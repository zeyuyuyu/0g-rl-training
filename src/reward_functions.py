"""
Reward Functions for veRL GRPO Training

veRL 调用 reward function 的方式有两种：
1. 同步: compute_score(data_source, solution_str, ground_truth, extra_info) -> dict
2. 异步: async compute_score(...) -> dict

本模块实现 execution-based reward function，参考:
- Qwen3.5-35B-A3B-Turbo-SWE 的 compile+run 方案
- veRL 文档: https://verl.readthedocs.io/en/latest/advance/reward_loop.html
"""

import re
import ast
import json
import tempfile
import subprocess
import signal
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# veRL interface: compute_score
#
# veRL 通过 reward.custom_reward_function.path 和 .name 加载此函数。
# 签名必须为: compute_score(data_source, solution_str, ground_truth, extra_info)
# 返回: {"score": float, ...}
# =============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    veRL reward function entry point.

    根据 data_source 选择不同的 reward 逻辑:
    - "coding" / "opencode" / "swe": execution-based reward
    - "agent":                       trajectory-based reward
    - 其他:                          fallback 到 format + length heuristic

    Args:
        data_source: 数据来源标识 (parquet 中的 data_source 字段)
        solution_str: 模型生成的完整 response 字符串
        ground_truth: reward_model.ground_truth 字段 (JSON string 或 plain text)
        extra_info: 额外信息 (split, index, test_cases 等)

    Returns:
        {"score": float} 其中 score 在 [0, 1] 范围内
    """
    extra_info = extra_info or {}

    if data_source in ("coding", "opencode", "swe", "code"):
        test_cases_raw = extra_info.get("test_cases")
        test_cases = _parse_test_cases(test_cases_raw, ground_truth)
        result = _code_execution_reward(solution_str, test_cases)
    elif data_source == "agent":
        result = _agent_trajectory_reward(solution_str, ground_truth, extra_info)
    else:
        result = _generic_reward(solution_str, ground_truth)

    return result


# =============================================================================
# Code Execution Reward
#
# R = 0.3 * compile + 0.5 * test + 0.2 * style
# 参考 Qwen3.5-35B-A3B-Turbo-SWE: execution-based (parse + compile + run)
# =============================================================================

COMPILE_WEIGHT = 0.3
TEST_WEIGHT = 0.5
STYLE_WEIGHT = 0.2
EXEC_TIMEOUT = 30  # seconds
MEMORY_LIMIT = 512 * 1024 * 1024  # 512 MB


def _code_execution_reward(
    solution_str: str,
    test_cases: List[Tuple[Any, Any]],
) -> Dict[str, Any]:
    """Execution-based reward: compile + run tests + style."""

    code = _extract_code(solution_str)
    if code is None:
        return {"score": 0.0, "compile": 0.0, "test": 0.0, "style": 0.0,
                "reason": "no_code_found"}

    # 1) Syntax / compile check
    syntax_ok, syntax_err = _check_syntax(code)
    compile_score = 1.0 if syntax_ok else 0.0

    # 2) Test execution (only if syntax OK)
    if syntax_ok and test_cases:
        test_score, passed, total = _run_tests(code, test_cases)
    elif syntax_ok:
        test_score, passed, total = 1.0, 0, 0
    else:
        test_score, passed, total = 0.0, 0, 0

    # 3) Style
    style_score = _check_style(code) if syntax_ok else 0.0

    total_score = (
        COMPILE_WEIGHT * compile_score
        + TEST_WEIGHT * test_score
        + STYLE_WEIGHT * style_score
    )

    return {
        "score": total_score,
        "compile": compile_score,
        "test": test_score,
        "style": style_score,
        "test_passed": passed,
        "test_total": total,
    }


def _extract_code(completion: str) -> Optional[str]:
    """Extract code from model output (```python blocks, raw code, etc.)."""
    patterns = [
        r'```python\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'<code>(.*?)</code>',
    ]
    for pat in patterns:
        matches = re.findall(pat, completion, re.DOTALL)
        if matches:
            return matches[-1].strip()

    if any(kw in completion for kw in ('def ', 'class ', 'import ', 'print(')):
        return completion.strip()
    return None


def _check_syntax(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def _run_tests(
    code: str,
    test_cases: List[Tuple[Any, Any]],
) -> Tuple[float, int, int]:
    """Run test cases in sandbox. Returns (pass_ratio, passed, total)."""
    if not test_cases:
        return 1.0, 0, 0

    passed = 0
    for test_input, expected in test_cases:
        ok, actual, _ = _execute_safely(code, test_input)
        if ok and _match_output(actual, expected):
            passed += 1

    total = len(test_cases)
    return passed / total, passed, total


def _execute_safely(
    code: str,
    test_input: Any = None,
) -> Tuple[bool, Any, str]:
    """Execute code in a subprocess sandbox with resource limits."""
    wrapped = (
        "import sys, json\n"
        "from io import StringIO\n"
        "_captured = StringIO()\n"
        "sys.stdout = _captured\n"
        f"{code}\n"
        "sys.stdout = sys.__stdout__\n"
        'print("__OUT__")\n'
        "print(json.dumps(_captured.getvalue()))\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapped)
        tmp = f.name

    try:
        result = subprocess.run(
            ['python3', tmp],
            capture_output=True, text=True,
            timeout=EXEC_TIMEOUT,
            preexec_fn=_limit_resources if hasattr(signal, 'SIGALRM') else None,
        )
        if result.returncode == 0:
            m = re.search(r'__OUT__\n(.+)$', result.stdout, re.DOTALL)
            if m:
                try:
                    return True, json.loads(m.group(1)), ""
                except json.JSONDecodeError:
                    return True, result.stdout, ""
            return True, result.stdout, ""
        return False, None, result.stderr
    except subprocess.TimeoutExpired:
        return False, None, f"timeout ({EXEC_TIMEOUT}s)"
    except Exception as e:
        return False, None, str(e)
    finally:
        import os
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _limit_resources():
    """Unix resource limits for sandbox."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def _match_output(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()
    try:
        if isinstance(actual, str):
            actual = json.loads(actual)
        if isinstance(expected, str):
            expected = json.loads(expected)
        return actual == expected
    except (json.JSONDecodeError, TypeError):
        pass
    return str(actual).strip() == str(expected).strip()


def _check_style(code: str) -> float:
    """Lightweight style score [0, 1]."""
    score = 1.0
    lines = code.split('\n')

    if 'def ' in code and '"""' not in code and "'''" not in code:
        score -= 0.1

    long_lines = sum(1 for l in lines if len(l) > 88)
    score -= min(0.1, long_lines * 0.01)

    import_re = re.compile(r'^(?:import |from )')
    found_body = False
    for line in lines:
        if re.search(r'^(?:def |class )', line):
            found_body = True
        if found_body and import_re.match(line.strip()):
            score -= 0.05
            break

    return max(0.0, score)


def _parse_test_cases(raw: Any, ground_truth: str) -> List[Tuple[Any, Any]]:
    """Parse test_cases from extra_info or ground_truth."""
    if raw:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if isinstance(raw, list):
            return [(tc.get("input"), tc.get("expected")) for tc in raw
                    if isinstance(tc, dict)]
    if ground_truth:
        try:
            gt = json.loads(ground_truth)
            if isinstance(gt, list):
                return [(tc.get("input"), tc.get("expected")) for tc in gt
                        if isinstance(tc, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


# =============================================================================
# Agent Trajectory Reward
# =============================================================================

def _agent_trajectory_reward(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reward for multi-turn agent traces.
    R = 0.5 * tool_success + 0.3 * task_completion + 0.2 * efficiency
    """
    try:
        trajectory = json.loads(solution_str) if isinstance(solution_str, str) else solution_str
    except (json.JSONDecodeError, TypeError):
        trajectory = []

    if not isinstance(trajectory, list):
        trajectory = []

    tool_turns = [t for t in trajectory if isinstance(t, dict) and t.get("role") == "tool"]
    if tool_turns:
        tool_ok = sum(1 for t in tool_turns if t.get("success", False))
        tool_rate = tool_ok / len(tool_turns)
    else:
        tool_rate = 0.5

    last_content = trajectory[-1].get("content", "") if trajectory else ""
    done_keywords = ("completed", "finished", "done", "success", "完成", "成功")
    task_done = 1.0 if any(kw in last_content.lower() for kw in done_keywords) else 0.0

    max_steps = extra_info.get("max_steps", 20)
    efficiency = max(0.0, 1.0 - len(trajectory) / max_steps)

    total = 0.5 * tool_rate + 0.3 * task_done + 0.2 * efficiency
    return {"score": total, "tool_success": tool_rate,
            "task_completion": task_done, "efficiency": efficiency}


# =============================================================================
# Generic / Fallback Reward
# =============================================================================

def _generic_reward(solution_str: str, ground_truth: str) -> Dict[str, Any]:
    """Fallback: exact-match + format heuristic."""
    if ground_truth and ground_truth.strip():
        if ground_truth.strip() in solution_str:
            return {"score": 1.0, "method": "exact_match"}

    score = 0.3
    if len(solution_str) > 50:
        score += 0.1
    if '```' in solution_str:
        score += 0.1
    if 'def ' in solution_str or 'class ' in solution_str:
        score += 0.1
    return {"score": min(1.0, score), "method": "heuristic"}
