# Reward Function 设计文档

## 1. 设计原则

Reward function 是 RL 训练中最关键的组件。一个好的 reward 需要满足：

| 原则 | 说明 | 坏例子 |
|-----|------|--------|
| **客观可验证** | 给定输入，reward 完全确定 | LLM-as-judge 每次打分不一样 |
| **抗 hack** | 模型不能通过 shortcut 获得高分 | 只看长度 → 模型学会输出废话 |
| **信号丰富** | 不只是 0/1，能给出 partial credit | 只看最终结果 → 梯度太稀疏 |
| **高效计算** | 不能成为训练瓶颈 | 人工标注 → 无法规模化 |

## 2. 代码生成 Reward (CodeExecutionReward)

### 2.1 总公式

```
R(completion) = w₁·Compile(code) + w₂·Test(code) + w₃·Style(code)
```

默认权重: `w₁=0.3, w₂=0.5, w₃=0.2`

### 2.2 各组件详解

#### Compile Score (权重 30%)

检查代码是否能通过 Python AST 解析。

```python
import ast

def check_compile(code: str) -> float:
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0
```

**设计考虑**:
- 为什么不用 `compile()`? 因为 `compile()` 会执行代码，不安全
- 为什么不用更严格的类型检查? 因为 Python 是动态类型，mypy-level 检查会过于严格
- partial credit? 不给。语法要么对要么错，没有中间状态

#### Test Score (权重 50%)

在沙箱中执行代码并运行测试用例。

```python
def compute_test_score(code: str, test_cases: list) -> float:
    passed = 0
    for test in test_cases:
        try:
            result = execute_in_sandbox(code, test["input"])
            if result == test["expected"]:
                passed += 1
        except (TimeoutError, MemoryError, Exception):
            pass
    return passed / max(len(test_cases), 1)
```

**沙箱配置**:
```python
SANDBOX_CONFIG = {
    "timeout": 10,           # 秒
    "memory_limit": 512,     # MB
    "network": False,        # 无网络
    "filesystem": "readonly",
    "max_processes": 1,
}
```

**Test cases 来源**:
1. **数据集自带**: 最可靠，每个 problem 附带 test cases
2. **LLM 生成**: 用 Claude/GPT 生成 test cases（需人工审核样本）
3. **Property-based**: 用 `hypothesis` 库自动生成边界条件
4. **编译测试**: 如果没有 test cases，只检查能否无 error 运行

#### Style Score (权重 20%)

检查代码风格，给予 partial credit。

```python
def check_style(code: str) -> float:
    score = 0.0
    checks = {
        "has_docstring": 0.4,      # 有 docstring
        "reasonable_length": 0.3,   # 行长度 < 120
        "imports_at_top": 0.2,      # import 在文件顶部
        "no_wildcard_import": 0.1,  # 无 from x import *
    }
    for check_name, weight in checks.items():
        if globals()[check_name](code):
            score += weight
    return score
```

**为什么需要 style 分数?**
- 防止模型生成"能跑但丑"的代码
- 鼓励可读性和可维护性
- 权重低 (20%)，不会过度约束创造性

### 2.3 Reward 分布分析

理想的 reward 分布应该是这样的：

```
训练初期:                      训练后期:
  │                              │
  │ ████                         │            ████
  │ ████████                     │         ████████
  │ ████████████                 │      ████████████
  │ ████████████████             │   ████████████████
  ──────────────────             ──────────────────
  0    0.25   0.5   0.75   1    0    0.25   0.5   0.75   1
  
  均值: ~0.35                     均值: ~0.65
  峰值在低分区                    峰值右移到高分区
```

**异常信号**:
- 二极化（大量 0 和 1，中间空）→ 任务太简单或太难
- 所有样本分数相同 → reward 信号无区分度
- 训练后分布没变 → 模型没在学

## 3. Agent Trajectory Reward (AgentTrajectoryReward)

用于多轮工具调用场景的 reward 设计。

### 3.1 总公式

```
R(trajectory) = w₁·ToolSuccess + w₂·TaskCompletion + w₃·Efficiency
```

默认权重: `w₁=0.5, w₂=0.3, w₃=0.2`

### 3.2 各组件详解

#### Tool Success Rate (权重 50%)

```python
def tool_success_rate(trajectory: list) -> float:
    tool_calls = [step for step in trajectory if step["type"] == "tool_call"]
    if not tool_calls:
        return 0.0
    successful = sum(1 for tc in tool_calls if tc["status"] == "success")
    return successful / len(tool_calls)
```

**关键细节**:
- 只计算模型主动发起的 tool call（不含系统自动调用的）
- "success" 定义: tool 返回非 error 结果
- 如果 trajectory 中无 tool call，返回 0 而非 1

#### Task Completion (权重 30%)

```python
def task_completion(trajectory: list, ground_truth: dict) -> float:
    final_state = trajectory[-1].get("state", {})
    return 1.0 if meets_requirements(final_state, ground_truth) else 0.0
```

**判定标准**:
- 代码类: 最终代码通过所有测试
- 文件操作: 目标文件存在且内容正确
- 信息检索: 返回正确答案

#### Efficiency (权重 20%)

```python
def efficiency_score(trajectory: list, max_steps: int = 20) -> float:
    actual_steps = len(trajectory)
    return max(0.0, 1.0 - actual_steps / max_steps)
```

**设计思路**:
- 鼓励模型用更少步骤完成任务
- `max_steps` 是预设的步数上限
- 超过 max_steps 的 trajectory，efficiency = 0

## 4. Reward Shaping 技巧

### 4.1 避免稀疏 reward

问题: 大部分 completion 的 reward 是 0（代码完全无法编译），梯度消失。

解决方案: 给 partial credit。

```python
def shaped_compile_score(code: str) -> float:
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError as e:
        lines = code.split('\n')
        error_line = e.lineno or len(lines)
        # 至少一部分代码是正确的
        return max(0.1, (error_line - 1) / len(lines) * 0.5)
```

### 4.2 防止 length exploitation

问题: 模型可能学会输出很长的代码以"碰运气"通过更多测试。

解决方案: 长度惩罚（仅对异常长的输出）。

```python
def length_penalty(code: str, expected_length: int = 200) -> float:
    actual = len(code.split('\n'))
    if actual <= expected_length * 2:
        return 1.0
    return max(0.5, 1.0 - (actual - expected_length * 2) / (expected_length * 10))
```

### 4.3 Reward 归一化

GRPO 内部会做 group normalization，但原始 reward 的 scale 也很重要：

```python
# 确保所有 reward 在 [0, 1] 范围内
def normalize_reward(raw_reward: float) -> float:
    return max(0.0, min(1.0, raw_reward))
```

## 5. Reward Hacking 案例与防护

| Hack 方式 | 描述 | 防护措施 |
|-----------|------|---------|
| **复制测试用例** | 模型直接输出 `assert` 语句 | 代码 vs 测试分离，禁止 `assert` 在主函数中 |
| **硬编码答案** | `return 42` | 使用随机化 test input |
| **无限循环 + try/except** | 包裹所有代码防止 crash | 超时终止 + 检查 try/except 覆盖率 |
| **空函数** | `pass` 通过编译 | 要求至少有 return 语句或 side effect |
| **字符串拼接** | 拼出正确输出而非计算 | 增加 test case 数量和多样性 |

## 6. 与 veRL 集成

veRL 通过 `reward.custom_reward_function` 配置加载自定义 reward function:

```bash
python3 -m verl.trainer.main_ppo \
    reward.custom_reward_function.path=./src/reward_functions.py \
    reward.custom_reward_function.name=compute_score
```

函数签名 (veRL 标准接口):

```python
def compute_score(
    data_source: str,       # parquet 中的 data_source 字段
    solution_str: str,      # 模型生成的 response
    ground_truth: str,      # reward_model.ground_truth
    extra_info: dict,       # extra_info 字段
) -> dict:                  # 必须包含 "score" key
    ...
    return {"score": 0.8, "compile": 1.0, "test": 0.7, "style": 0.9}
```

veRL 支持同步和异步两种模式。异步模式适合需要沙箱执行的场景:

```python
async def compute_score(data_source, solution_str, ground_truth, extra_info):
    result = await run_code_in_sandbox(solution_str)
    return {"score": result.pass_rate}
```

参考: [veRL Reward Loop 文档](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)

## 7. 未来方向

1. **混合 Reward**: Execution-based (70%) + SWE-RM model-based (30%)
2. **Self-play**: 模型互相生成 test cases
3. **Process Reward**: 不只奖励最终结果，也奖励正确的中间推理步骤
4. **Online Reward Adaptation**: 根据训练阶段动态调整 reward 权重
