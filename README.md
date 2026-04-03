# RL Training for 0G Platform

基于 [veRL](https://verl.readthedocs.io/) 框架的 RL 训练代码，采用 **GRPO (Group Relative Policy Optimization)** 算法，专为 0G Compute Network 的长上下文（256K）Agentic AI 模型设计。

> **参考实现**: [Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) 的 SFT+GRPO 训练流程

---

## 设计目标

```
1. 长上下文 (256K) → 渐进式训练策略
2. MoE 模型稳定性 → Router 冻结 + 低 LR
3. Agentic 能力提升 → Execution-based Reward
4. 0G 平台集成 → 自动 Hash 计算 + 配置生成
```

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RL Training Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐ │
│  │   Data Layer    │        │  Training Core  │        │   0G Export     │ │
│  │                 │───────>│                 │───────>│                 │ │
│  │ Session Extract │        │ veRL main_ppo   │        │ LoRA Merge      │ │
│  │ Quality Score   │        │ GRPO algorithm  │        │ Keccak256 Hash  │ │
│  │ veRL Parquet    │        │ vLLM Rollout    │        │ Config Gen      │ │
│  └─────────────────┘        └─────────────────┘        └─────────────────┘ │
│                                                                             │
│  Input:                     veRL CLI:                  Output:              │
│  5-field parquet            python3 -m verl            model_hash.txt       │
│  (data_source,              .trainer.main_ppo          0g_config.json       │
│   prompt,                   algorithm.adv_estimator    SCRIPT_MAP entry     │
│   ability,                  =grpo                                           │
│   reward_model,                                                             │
│   extra_info)                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 完整训练流程

### Phase 1: 数据准备

将原始 agent sessions 转化为 veRL 要求的 5 字段 Parquet 格式。

```bash
python src/data_processor.py \
    --input ./raw/sessions.jsonl \
    --output-dir ./data \
    --format opencode \
    --min-quality 0.4
```

veRL Parquet Schema (5 required fields):

| 字段 | 类型 | 说明 |
|-----|------|------|
| `data_source` | str | 数据来源标识，RewardManager 用此选择 reward function |
| `prompt` | list[dict] | HuggingFace chat template: `[{"role":"user","content":"..."}]` |
| `ability` | str | 任务类别: `"coding"`, `"agent"` |
| `reward_model` | dict | `{"style":"rule","ground_truth":"..."}` |
| `extra_info` | dict | 附加元数据 (split, index, test_cases 等) |

---

### Phase 2: SFT

```
Dataset: 2,674 high-quality pairs (score > 0.6)
Config:  bf16, LoRA r=64, 670 steps
Loss:    1.438 → 0.509 (-65%)
Purpose: 建立基础 coding capability，为 RL 提供 init checkpoint
```

---

### Phase 3: GRPO RL 训练

核心: 使用 veRL 的 `main_ppo` 入口，通过 `algorithm.adv_estimator=grpo` 启用 GRPO。

#### Why GRPO?

| | PPO | GRPO |
|---|-----|------|
| Critic Model | 需要 (与 actor 同规模) | 不需要 |
| 显存 | 2× actor | ~1.2× actor |
| 训练成本 | baseline | ~1/18 |
| Advantage | R - V(s), 需训练 V | (rᵢ - group_mean) / group_std |

#### 训练命令

```bash
# 使用我们封装的训练脚本 (内部调用 veRL CLI)
./scripts/train_grpo.sh --gpus 4 --model Qwen/Qwen3.5-35B-A3B

# 本地快速测试
./scripts/train_grpo.sh --local-test

# 或直接调用 veRL CLI
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/train.parquet \
    data.val_files=./data/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=32768 \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-35B-A3B \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    reward.custom_reward_function.path=./src/reward_functions.py \
    reward.custom_reward_function.name=compute_score \
    trainer.total_epochs=5
```

#### 核心参数说明

| veRL 参数 | 值 | 说明 |
|-----------|-----|------|
| `algorithm.adv_estimator` | `grpo` | 关键开关：启用 GRPO |
| `actor_rollout_ref.rollout.n` | `8` | Group size G=8 (Qwen3.5-SWE 同款) |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | GRPO: KL 加到 loss (非 reward) |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | KL 惩罚系数 |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | 低方差 KL 估计 |
| `actor_rollout_ref.actor.optim.lr` | `5e-6` | MoE 用更小 LR (dense 用 1e-5) |
| `algorithm.use_kl_in_reward` | `False` | GRPO 将 KL 加到 loss 而非 reward |
| `trainer.critic_warmup` | `0` | GRPO 无 critic |

---

### Phase 4: 长上下文渐进式扩展 (256K)

```
Step 0-500:      32K  tokens  (batch=128)
Step 500-1000:   64K  tokens  (batch=64)
Step 1000-1500:  128K tokens  (batch=32)
Step 1500+:      256K tokens  (batch=16)
```

通过调整 `data.max_response_length` 和 `data.train_batch_size` 实现。

---

## Reward Function 设计

Execution-based reward，防止 reward hacking。

### veRL 集成方式

veRL 通过以下配置加载自定义 reward function:

```bash
reward.custom_reward_function.path=./src/reward_functions.py
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

### Reward 组成

```
R(solution) = 0.3 × Compile + 0.5 × Test + 0.2 × Style

Compile: Python AST parse (0 or 1)
Test:    沙箱执行 + test case 对比 ([0, 1])
Style:   Docstring / imports / line length ([0, 1])
```

根据 `data_source` 自动选择 reward 逻辑:
- `"coding"` / `"opencode"` / `"swe"` → execution-based
- `"agent"` → trajectory-based (tool success + task completion + efficiency)
- 其他 → exact match + heuristic fallback

---

## 快速开始

### 1. 环境安装

```bash
conda create -n verl python=3.10
conda activate verl

pip install -r requirements.txt
pip install verl
pip install vllm==0.8.2
pip install pycryptodome
```

### 2. 数据准备

```bash
python src/data_processor.py \
    --input ./raw/opencode_sessions.jsonl \
    --output-dir ./data \
    --format opencode \
    --min-quality 0.4

# 验证 parquet schema
python -c "
import pyarrow.parquet as pq
t = pq.read_table('./data/train.parquet')
print(t.column_names)  # 应包含: data_source, prompt, ability, reward_model, extra_info
"
```

### 3. 启动训练

```bash
# 本地测试 (0.5B 模型, 1 GPU, 1 epoch)
./scripts/train_grpo.sh --local-test

# 正式训练
./scripts/train_grpo.sh --gpus 4 --model Qwen/Qwen3.5-35B-A3B --epochs 5
```

### 4. 导出到 0G 平台

```bash
python scripts/export_for_0g.py \
    --checkpoint ./checkpoints/grpo/final \
    --output ./models/0g_ready \
    --compute-hash \
    --generate-config
```

---

## 预期效果

参考 [Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1):

| 阶段 | 参数 | 硬件 |
|-----|------|------|
| SFT | Loss: 1.438→0.509, 670 steps, ~3h | RTX PRO 6000 96GB |
| GRPO Sampling | 200 prompts × G=8 = 1,600 completions, ~2h | Same |
| GRPO Update | RFT on 161 best (reward>=0.5), 123 steps, ~12min | Same |

| 指标 | SFT 后 | GRPO 后 | 提升 |
|-----|--------|---------|------|
| Compile Success | ~60% | ~85% | +25% |
| Test Pass | ~40% | ~75% | +35% |
| Avg Reward | 0.35 | 0.65 | +86% |

---

## 0G 平台集成

### 添加模型到 SCRIPT_MAP

```go
// api/fine-tuning/const/const.go
SCRIPT_MAP = map[string]ModelConfig{
    "0x[model_hash]": {
        TrainingScript:   "/app/train_lora.py",
        PriceCoefficient: 4,
        StorageFee:       15000000000000000,
    },
}
```

### 配置 user_config.yaml

```yaml
ModelLocalPaths:
  "0x[hash]": "/dstack/persistent/models/your-rl-model"
```

---

## 监控与调试

```bash
# TensorBoard
tensorboard --logdir ./logs

# 关键 veRL metrics:
# - actor/loss: 应逐渐下降
# - actor/kl_loss: < 0.1
# - reward/mean: 应逐渐上升
```

```bash
# 模型评估
python src/evaluate.py \
    --model ./checkpoints/grpo/final \
    --test-data ./data/val.jsonl \
    --use-vllm
```

---

## 核心设计决策总结

| 决策 | 选择 | 理由 |
|-----|------|------|
| RL 算法 | GRPO | 无 critic，省显存，成本 ~1/18 PPO |
| 训练框架 | veRL | 原生 GRPO 支持，FSDP+vLLM 融合 |
| Reward | Execution-based | 客观可验证，防 reward hacking |
| veRL 入口 | `main_ppo` + CLI | 官方标准接口，参数用 `actor_rollout_ref.*` |
| 数据格式 | 5-field Parquet | veRL RLHFDataset 标准 schema |
| MoE 优化 | 冻结 router + lr=5e-6 | 防止 expert 崩溃 |
| 导出格式 | LoRA merge + keccak256 | 0G 平台兼容 |

---

## 参考资源

- [veRL Documentation](https://verl.readthedocs.io/)
- [veRL GRPO Algorithm](https://verl.readthedocs.io/en/v0.5.x/algo/grpo.html)
- [veRL Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)
- [veRL Data Preparation](https://verl.readthedocs.io/en/v0.5.x/preparation/prepare_data.html)
- [veRL GRPO Examples](https://github.com/verl-project/verl/tree/main/examples/grpo_trainer)
- [Qwen3.5-35B-A3B-Turbo-SWE](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1)
- [DeepSeekMath Paper (GRPO)](https://arxiv.org/pdf/2402.03300)

---

## License

Apache 2.0
