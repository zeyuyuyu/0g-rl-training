"""
Reward Functions for RL Training

This module implements execution-based reward functions for code generation tasks,
reference: https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1

The reward function is the core of RL training - it evaluates the quality of
generated code by actually executing it.
"""

import re
import ast
import tempfile
import subprocess
import signal
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import traceback


@dataclass
class RewardResult:
    """Result of reward computation"""
    total_score: float
    breakdown: Dict[str, float]
    details: Dict[str, Any]
    execution_output: str = ""
    error_message: str = ""


class CodeExecutionReward:
    """
    Execution-based reward function for code generation.
    
    Similar to the one used in Qwen3.5-35B-A3B-Turbo-SWE training:
    - Compile/parse check: 30%
    - Test execution: 50%
    - Code style: 20%
    
    Reference: https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1
    """
    
    def __init__(
        self,
        compile_weight: float = 0.3,
        test_weight: float = 0.5,
        style_weight: float = 0.2,
        timeout: int = 30,
        test_cases: Optional[List[Tuple[Any, Any]]] = None
    ):
        """
        Args:
            compile_weight: Weight for compilation success
            test_weight: Weight for test execution
            style_weight: Weight for code style
            timeout: Execution timeout in seconds
            test_cases: List of (input, expected_output) tuples
        """
        self.compile_weight = compile_weight
        self.test_weight = test_weight
        self.style_weight = style_weight
        self.timeout = timeout
        self.test_cases = test_cases or []
        
        # Verify weights sum to 1.0
        total = compile_weight + test_weight + style_weight
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
    
    def extract_code(self, completion: str) -> Optional[str]:
        """
        Extract code from model completion.
        Handles various formats:
        - Code blocks with ```python or ```
        - Raw code
        - With/without explanation text
        """
        # Try to extract from code blocks
        patterns = [
            r'```python\n(.*?)\n```',  # ```python
            r'```\n(.*?)\n```',         # ```
            r'<code>(.*?)</code>',      # <code> tags
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, completion, re.DOTALL)
            if matches:
                return matches[-1].strip()  # Return last match
        
        # If no code blocks, try to parse the whole thing
        # Heuristic: look for common Python constructs
        if any(keyword in completion for keyword in ['def ', 'class ', 'import ', 'print(']):
            return completion.strip()
        
        return None
    
    def check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    def execute_code_safely(
        self,
        code: str,
        test_input: Any = None
    ) -> Tuple[bool, Any, str]:
        """
        Execute code in a sandboxed environment.
        
        Returns:
            (success, output, error_message)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code to capture output
            wrapped_code = f"""
import sys
import json

# Redirect stdout to capture output
from io import StringIO
captured_output = StringIO()
sys.stdout = captured_output

# User code
{code}

# Restore stdout
sys.stdout = sys.__stdout__

# Print captured output as JSON
print("\\n__CAPTURED_OUTPUT__")
print(json.dumps(captured_output.getvalue()))
"""
            f.write(wrapped_code)
            temp_file = f.name
        
        try:
            # Run with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # Security: limit resources
                preexec_fn=self._limit_resources if hasattr(signal, 'SIGALRM') else None
            )
            
            if result.returncode == 0:
                # Extract captured output
                output_match = re.search(
                    r'__CAPTURED_OUTPUT__\n(.+)$',
                    result.stdout,
                    re.DOTALL
                )
                if output_match:
                    import json
                    try:
                        output = json.loads(output_match.group(1))
                    except:
                        output = result.stdout
                else:
                    output = result.stdout
                
                return True, output, ""
            else:
                return False, None, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, None, f"Execution timeout after {self.timeout}s"
        except Exception as e:
            return False, None, str(e)
        finally:
            import os
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _limit_resources(self):
        """Limit system resources for sandbox (Unix only)"""
        import resource
        # Limit memory to 512MB
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        # Limit CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    
    def run_tests(self, code: str) -> Tuple[float, int, int, str]:
        """
        Run test cases against the code.
        
        Returns:
            (pass_ratio, passed_count, total_count, error_message)
        """
        if not self.test_cases:
            # No tests provided, skip
            return 1.0, 0, 0, ""
        
        passed = 0
        errors = []
        
        for test_input, expected_output in self.test_cases:
            success, actual_output, error = self.execute_code_safely(
                code,
                test_input
            )
            
            if success:
                # Compare output (flexible matching)
                if self._match_output(actual_output, expected_output):
                    passed += 1
                else:
                    errors.append(f"Test failed: expected {expected_output}, got {actual_output}")
            else:
                errors.append(f"Test error: {error}")
        
        total = len(self.test_cases)
        ratio = passed / total if total > 0 else 1.0
        
        return ratio, passed, total, "; ".join(errors) if errors else ""
    
    def _match_output(self, actual: Any, expected: Any) -> bool:
        """Flexible output matching"""
        # String match
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.strip() == actual.strip()
        
        # JSON/structure match
        try:
            import json
            if isinstance(actual, str):
                actual = json.loads(actual)
            if isinstance(expected, str):
                expected = json.loads(expected)
            return actual == expected
        except:
            pass
        
        # Direct comparison
        return actual == expected
    
    def check_style(self, code: str) -> float:
        """
        Check code style quality.
        Returns score between 0 and 1.
        """
        score = 1.0
        deductions = []
        
        # Check for basic Python style guidelines
        lines = code.split('\n')
        
        # 1. Function/class docstrings
        if 'def ' in code and '"""' not in code and "'''" not in code:
            deductions.append(0.1)  # Missing docstrings
        
        # 2. Line length
        long_lines = sum(1 for line in lines if len(line) > 88)
        if long_lines > 0:
            deductions.append(min(0.1, long_lines * 0.01))
        
        # 3. Imports (should be at top)
        import_pattern = re.compile(r'^import |^from ')
        non_top_imports = 0
        found_code = False
        for line in lines:
            if re.search(r'def |class ', line):
                found_code = True
            if found_code and import_pattern.match(line.strip()):
                non_top_imports += 1
        if non_top_imports > 0:
            deductions.append(0.1)
        
        # 4. Check for meaningful variable names (heuristic)
        single_char_vars = len(re.findall(r'\b[abcxyzij]\b', code))
        if single_char_vars > 5:
            deductions.append(0.05)
        
        # Apply deductions
        score -= sum(deductions)
        return max(0.0, score)
    
    def compute_reward(self, completion: str) -> RewardResult:
        """
        Main entry point: compute reward for a completion.
        """
        # Step 1: Extract code
        code = self.extract_code(completion)
        if code is None:
            return RewardResult(
                total_score=0.0,
                breakdown={"extract": 0.0},
                details={"error": "No code found in completion"},
                error_message="Failed to extract code"
            )
        
        # Step 2: Syntax check (30% of score)
        syntax_ok, syntax_error = self.check_syntax(code)
        compile_score = 1.0 if syntax_ok else 0.0
        
        # Step 3: Test execution (50% of score)
        if syntax_ok:
            test_ratio, passed, total, test_error = self.run_tests(code)
            test_score = test_ratio
        else:
            test_score = 0.0
            test_error = "Syntax error, skipping tests"
        
        # Step 4: Style check (20% of score)
        style_score = self.check_style(code)
        
        # Calculate weighted total
        total_score = (
            self.compile_weight * compile_score +
            self.test_weight * test_score +
            self.style_weight * style_score
        )
        
        return RewardResult(
            total_score=total_score,
            breakdown={
                "compile": compile_score,
                "test": test_score,
                "style": style_score
            },
            details={
                "syntax_ok": syntax_ok,
                "syntax_error": syntax_error if not syntax_ok else "",
                "test_passed": passed if syntax_ok else 0,
                "test_total": total if syntax_ok else 0,
                "test_error": test_error
            },
            error_message=""
        )


class AgentTrajectoryReward:
    """
    Reward function for general agent trajectories (not just code).
    
    Suitable for tool-use agents, multi-turn conversations, etc.
    """
    
    def __init__(
        self,
        tool_success_weight: float = 0.5,
        task_completion_weight: float = 0.3,
        efficiency_weight: float = 0.2,
        max_steps: int = 20
    ):
        self.tool_success_weight = tool_success_weight
        self.task_completion_weight = task_completion_weight
        self.efficiency_weight = efficiency_weight
        self.max_steps = max_steps
    
    def compute_reward(self, trajectory: List[Dict]) -> RewardResult:
        """
        Compute reward for a multi-turn agent trajectory.
        
        Args:
            trajectory: List of turns, each with {
                "role": "assistant" | "tool",
                "content": str,
                "tool_calls": List[Dict],  # for assistant
                "success": bool,  # for tool
            }
        """
        if not trajectory:
            return RewardResult(0.0, {}, {}, error_message="Empty trajectory")
        
        # 1. Tool success rate
        tool_turns = [t for t in trajectory if t.get("role") == "tool"]
        if tool_turns:
            success_count = sum(1 for t in tool_turns if t.get("success", False))
            tool_success_rate = success_count / len(tool_turns)
        else:
            tool_success_rate = 1.0  # No tools used, neutral
        
        # 2. Task completion (requires external judge)
        # For now, use heuristics
        last_message = trajectory[-1].get("content", "")
        completion_indicators = [
            "completed", "finished", "done", "success",
            "结果", "完成", "成功"
        ]
        task_completion = any(ind in last_message.lower() for ind in completion_indicators)
        task_score = 1.0 if task_completion else 0.0
        
        # 3. Efficiency (fewer steps is better)
        num_steps = len(trajectory)
        efficiency = max(0.0, 1.0 - (num_steps / self.max_steps))
        
        # Weighted total
        total = (
            self.tool_success_weight * tool_success_rate +
            self.task_completion_weight * task_score +
            self.efficiency_weight * efficiency
        )
        
        return RewardResult(
            total_score=total,
            breakdown={
                "tool_success": tool_success_rate,
                "task_completion": task_score,
                "efficiency": efficiency
            },
            details={
                "num_steps": num_steps,
                "num_tools": len(tool_turns)
            }
        )


# Factory function for veRL integration
def get_reward_function(config: Dict[str, Any]) -> callable:
    """
    Factory to create reward function based on config.
    This is the entry point veRL will use.
    """
    reward_type = config.get("type", "code_execution")
    
    if reward_type == "code_execution":
        reward_fn = CodeExecutionReward(
            compile_weight=config.get("compile_weight", 0.3),
            test_weight=config.get("test_weight", 0.5),
            style_weight=config.get("style_weight", 0.2),
            timeout=config.get("timeout", 30),
            test_cases=config.get("test_cases", [])
        )
    elif reward_type == "agent_trajectory":
        reward_fn = AgentTrajectoryReward(
            tool_success_weight=config.get("tool_success_weight", 0.5),
            task_completion_weight=config.get("task_completion_weight", 0.3),
            efficiency_weight=config.get("efficiency_weight", 0.2)
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    # Return a callable that veRL can use
    def reward_wrapper(completions: List[str], **kwargs) -> List[float]:
        """Wrapper for batch processing"""
        rewards = []
        for completion in completions:
            result = reward_fn.compute_reward(completion)
            rewards.append(result.total_score)
        return rewards
    
    return reward_wrapper


# Default reward function for veRL config
code_execution_reward = get_reward_function({
    "type": "code_execution",
    "compile_weight": 0.3,
    "test_weight": 0.5,
    "style_weight": 0.2,
    "timeout": 30
})
