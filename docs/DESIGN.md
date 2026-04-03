# RL Training Design Document

## 1. 概述

本项目为 0G Compute Network 提供基于 GRPO (Group Relative Policy Optimization) 的 RL 训练能力，使 fine-tuned 模型具备更强的 agentic 能力——包括多轮工具调用、长上下文推理和代码生成。

### 1.1 背景与动机

传统 SFT 的局限性：
- **模式模仿 (Imitation)**: SFT 只学习"怎么做"，不学习"什么是好的"
- **分布偏移 (Distribution Shift)**: 推理时遇到训练数据外的情况，模型容易崩
- **缺乏自我纠错**: SFT 不具备在多轮交互中从错误中学习的能力

RL 的核心价值：
- **探索 + 反馈循环**: 模型自主生成多种方案，通过 reward 信号学习哪个更好
- **泛化能力**: 不再依赖特定的 (input, output) pair，而是学会通用的问题解决策略
- **自我提升**: 可以超越训练数据本身的质量上限

### 1.2 训练范式演进

```
2022-2023: SFT → RLHF (PPO + Reward Model)
    ↓
2024:      SFT → DPO/ORPO (无需 RL，pair-wise 偏好)
    ↓
2025:      SFT → GRPO (DeepSeek-R1，无需 critic)
    ↓
2026:      SFT → DAPO/iGRPO (更稳定的长序列 RL)
```

参考来源:
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [From PPO to GRPO to DAPO](https://softmaxdata.com/blog/from-ppo-to-grpo-to-dapo-understanding-rl-for-llms-and-every-training-parameter-explained/)

---

## 2. 算法选型

### 2.1 为什么选择 GRPO

| 算法 | Critic | 显存开销 | 训练成本 | 长序列稳定性 | 适用场景 |
|------|--------|---------|---------|-------------|---------|
| **PPO** | 需要独立 value model | 2× actor | 高 | 一般 | 通用 RL |
| **DPO** | 不需要 | 1× actor | 低 | 好 | 偏好对齐（非推理） |
| **GRPO** | 不需要 | ~1.2× actor | ~1/18 PPO | 好 | 推理 / 代码生成 |
| **DAPO** | 不需要 | ~1.2× actor | 低 | 最好 | 长 CoT / Agent |
| **iGRPO** | 不需要 | ~1.5× actor | 中 | 好 | 自我迭代 |

GRPO 的核心优势：
1. **无 Critic**: 用 group 内的 mean/std 替代 value function，省掉一个与 actor 同规模的模型
2. **采样高效**: 每个 prompt 生成 G 个 response，天然适合 vLLM 的 batch inference
3. **生态成熟**: veRL、TRL、Open-Instruct 均已集成，有大量 baseline 可参考

### 2.2 GRPO 数学原理

给定 prompt x，策略 π_θ 生成 G 个 response {y₁, y₂, ..., y_G}：

```
1. 计算 reward:  rᵢ = R(x, yᵢ)

2. Group-relative advantage:
   Â(x, yᵢ) = (rᵢ - mean({r₁,...,r_G})) / std({r₁,...,r_G})

3. Clipped policy gradient (类似 PPO):
   L_GRPO = -E [ min(ρᵢ·Âᵢ, clip(ρᵢ, 1-ε, 1+ε)·Âᵢ) ]
   
   其中 ρᵢ = π_θ(yᵢ|x) / π_θ_old(yᵢ|x)

4. KL regularization (加到 loss 而非 reward):
   L_total = L_GRPO + β · D_KL(π_θ || π_ref)
```

**与 PPO 的关键区别**：PPO 的 advantage 是 `A = R - V(s)` 需要训练 value function V(s)；GRPO 用 group mean 替代 V(s)，当 G 足够大时，group mean 是 V(s) 的无偏估计。

### 2.3 GRPO 变体与演进

**DrGRPO** (2025): 解决 GRPO 在长序列中的 length bias——GRPO 的 group normalization 会导致错误的长输出获得人为偏高的 advantage。DrGRPO 改为全局常数归一化 + 移除 KL loss + 关闭 std normalization。

```bash
# DrGRPO veRL 配置
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_kl_loss=false \
    algorithm.norm_adv_by_std_in_grpo=false
```

**DAPO** (ByteDance, 2025): 四个关键改进：
- **Clip-Higher**: 非对称裁剪 (ε⁺=0.28, ε⁻=0.2)，保持探索能力
- **Dynamic Sampling**: 过滤 reward 方差为 0 的 batch，避免梯度消失
- **Token-level Policy Gradient**: 防止长序列中的梯度坍塌
- **Overlong Reward Shaping**: 对超长输出施加平滑惩罚

在 AIME 2024 上，DAPO 训练的 Qwen2.5-32B 达到 50 分，比 DeepSeek-R1-Zero 用少 50% 的步数。

**iGRPO** (2026): 两阶段自我反馈：
- Stage 1: 生成探索性草稿并选择最佳
- Stage 2: 基于草稿条件进行 GRPO 更新
- AIME24 上达到 85.62%

**我们的选择**: Phase 1 先用标准 GRPO 跑通 pipeline，Phase 2 切换到 DAPO 以获得更好的长序列稳定性。

---

## 3. veRL 框架集成

### 3.1 veRL 架构

[veRL](https://verl.readthedocs.io/) 是 HybridFlow paper 的开源实现，核心特点：

- **Actor-Rollout 融合**: 3D-HybridEngine 消除 actor 和 rollout 之间的模型冗余
- **多后端支持**: PyTorch FSDP / Megatron-LM (training) + vLLM / SGLang (inference)
- **算法支持**: PPO, GRPO, DrGRPO, REINFORCE++

#### 3.1.1 veRL 核心角色模型

```python
from verl.trainer.ppo.ray_trainer import Role

# veRL 定义的角色:
# Role.ActorRollout      — actor + rollout 融合 (HybridEngine)
# Role.Critic            — value function (PPO 需要, GRPO 不需要)
# Role.RefPolicy         — reference policy (KL 约束)
# Role.RewardModel       — model-based reward (可选)
# Role.ActorRolloutRef   — actor + rollout + ref 全融合
```

GRPO 的关键优势: 不需要 `Role.Critic`，节省一整个模型的显存。

#### 3.1.2 Worker 与 Resource Pool

veRL 使用 Ray 做分布式编排，所有模型共享同一组 GPU (co-locate):

```python
role_worker_mapping = {
    Role.ActorRollout: ActorRolloutRefWorker,   # FSDP or Megatron
    Role.RefPolicy:    ActorRolloutRefWorker,
    # Role.Critic: 不需要 (GRPO)
}

resource_pool_spec = {
    'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
```

#### 3.1.3 训练入口

veRL 使用 Hydra/OmegaConf 管理配置，训练入口是:

```bash
python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
```

所有参数通过 dot-notation 覆盖。**注意参数前缀是 `actor_rollout_ref`（不是 `actor_rollout`）**。

### 3.2 数据层 — veRL Parquet Schema

veRL 的 `RLHFDataset` 要求 parquet 文件包含 5 个标准字段:

| 字段 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `data_source` | str | 数据来源标识，用于 RewardManager 选择 reward function | `"coding"`, `"openai/gsm8k"` |
| `prompt` | list[dict] | HuggingFace chat template 格式 | `[{"role":"user","content":"..."}]` |
| `ability` | str | 任务类别 | `"coding"`, `"agent"`, `"math"` |
| `reward_model` | dict | reward 配置 | `{"style":"rule","ground_truth":"42"}` |
| `extra_info` | dict | 附加元数据 | `{"split":"train","index":0,"test_cases":"..."}` |

#### 数据转换流程

```python
# 遵循 veRL data_preprocess 规范
def make_map_fn(split):
    def process_fn(example, idx):
        return {
            "data_source": "coding",
            "prompt": [{"role": "user", "content": example["instruction"]}],
            "ability": "coding",
            "reward_model": {"style": "rule", "ground_truth": "..."},
            "extra_info": {"split": split, "index": idx},
        }
    return process_fn

dataset = dataset.map(function=make_map_fn("train"), with_indices=True)
dataset.to_parquet("train.parquet")
```

### 3.3 Reward Function — veRL compute_score 接口

veRL 通过配置加载自定义 reward function:

```bash
reward.custom_reward_function.path=./src/reward_functions.py
reward.custom_reward_function.name=compute_score
```

函数签名必须为:

```python
def compute_score(
    data_source: str,       # parquet 中的 data_source
    solution_str: str,      # 模型生成的 response
    ground_truth: str,      # reward_model.ground_truth
    extra_info: dict,       # extra_info 字段
) -> dict:                  # 必须包含 "score" key
    ...
    return {"score": 0.8}
```

veRL 支持同步和异步两种模式。对于需要沙箱执行代码的场景，异步模式效率更高:

```python
async def compute_score(data_source, solution_str, ground_truth, extra_info):
    result = await run_code_in_sandbox(solution_str)
    return {"score": result.pass_rate}
```

#### Distributed Reward 架构

veRL v0.5+ 引入了 Reward Loop 架构，支持:
- **RewardLoopManager**: 将 batch 分发到多个 RewardWorker 并行计算
- **Streaming Reward**: rollout 生成一个样本就立即计算 reward，不需要等全部完成
- **Hybrid Reward**: 同一 pipeline 可以混合 rule-based + model-based reward

```python
class RewardLoopManager:
    def compute_rm_score(self, data):
        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get([
            worker.compute_score_batch.remote(chunk)
            for worker, chunk in zip(self.reward_loop_workers, chunks)
        ])
        return outputs
```

### 3.4 GRPO 关键配置参数

以下参数直接对应 veRL CLI 的 dot-notation keys:

| veRL 参数 | 值 | 说明 |
|-----------|-----|------|
| `algorithm.adv_estimator` | `grpo` | 启用 GRPO (关键开关) |
| `algorithm.use_kl_in_reward` | `False` | GRPO 将 KL 加到 loss (非 reward) |
| `actor_rollout_ref.rollout.n` | `8` | Group size (G=8, 每 prompt 8 completions) |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | 启用 KL loss |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | KL 惩罚系数 |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | 低方差 KL 估计 (k3 estimator) |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` | Token-level 聚合 (长序列更稳定) |
| `actor_rollout_ref.actor.optim.lr` | `5e-6` | MoE 模型 LR (dense 用 1e-5) |
| `actor_rollout_ref.rollout.name` | `vllm` | Rollout 引擎 (vllm / sglang) |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `4` | TP 并行度 |
| `trainer.critic_warmup` | `0` | GRPO 无 critic, 设为 0 |

#### 与 veRL 官方示例的对比

我们的配置基于 `examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh` 修改:

```diff
  # veRL 官方 GSM8K 示例 (7B dense)
  actor_rollout_ref.actor.optim.lr=5e-8
  actor_rollout_ref.rollout.n=5
  data.train_batch_size=1024

  # 我们的配置 (35B MoE, coding)
+ actor_rollout_ref.actor.optim.lr=5e-6     # MoE 需要更大 LR
+ actor_rollout_ref.rollout.n=8              # 参考 Qwen3.5-SWE 的 G=8
+ data.train_batch_size=128                  # 35B 模型 batch 更小
+ data.max_response_length=32768             # 长 CoT 需要更大 response
+ reward.custom_reward_function.path=...     # 自定义 execution-based reward
```

---

## 4. 训练架构

### 4.1 端到端流程

```
Phase 1                Phase 2              Phase 3              Phase 4
数据准备                SFT                  GRPO RL              导出部署
─────────────────────────────────────────────────────────────────────────

Agent Sessions          High-Quality         SFT Checkpoint       RL Checkpoint
(Codex/Claude/         Pairs                     │                     │
 OpenCode)              (score>0.6)               │                     │
     │                      │                     ▼                     ▼
     ▼                      ▼               ┌───────────┐        ┌───────────┐
┌──────────┐          ┌──────────┐          │ For each  │        │ LoRA      │
│ Extract  │          │ LoRA SFT │          │ prompt:   │        │ Merge     │
│ & Filter │          │ r=64     │          │           │        │     │     │
│ & Score  │          │ bf16     │          │ Sample G=8│        │ Keccak256 │
└──────────┘          └──────────┘          │ responses │        │ Hash      │
     │                      │               │     │     │        │     │     │
     ▼                      ▼               │ Compute   │        │ 0G Config │
  Parquet              SFT Model            │ Reward    │        │ Generate  │
  Dataset              Checkpoint           │     │     │        └───────────┘
                                            │ GRPO     │              │
  4,756 sessions       Loss: 1.438→0.509    │ Update    │              ▼
  → 3,580 pairs        670 steps, ~3h       └───────────┘        Deploy to 0G
                                            ~2h sampling
                                            ~12min update
```

### 4.2 Reward Function 设计

这是 RL 训练中最关键的组件。设计不好的 reward 会导致 reward hacking。

#### 为什么选择 Execution-based Reward

| 方法 | 客观性 | 可 hack | 成本 | 适用 |
|-----|--------|---------|------|------|
| Human feedback | 主观 | 难 | 极高 | 通用对齐 |
| LLM-as-judge | 中 | 容易 | 中 | 开放式生成 |
| Rule-based | 客观 | 不可能 | 低 | 格式检查 |
| **Execution-based** | **客观** | **极难** | **中** | **代码生成** |

最新研究 [SWE-RM](https://openreview.net/pdf/f1a199e02ff70ac9a67394a0d3aa7cf82d9df118.pdf) 提出了 MoE-based reward model (30B total, 3B activated) 用于不依赖 unit test 的场景，作为 execution-based reward 的补充。

#### Reward 实现

我们的 `compute_score` 函数根据 `data_source` 自动选择 reward 逻辑:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    if data_source in ("coding", "opencode", "swe"):
        # R = 0.3×Compile + 0.5×Test + 0.2×Style
        return _code_execution_reward(solution_str, test_cases)
    elif data_source == "agent":
        # R = 0.5×ToolSuccess + 0.3×TaskCompletion + 0.2×Efficiency
        return _agent_trajectory_reward(solution_str, ground_truth, extra_info)
    else:
        # Fallback: exact match + heuristic
        return _generic_reward(solution_str, ground_truth)
```

#### 沙箱执行安全

代码在隔离的子进程中执行:
- 内存限制: 512MB (`resource.RLIMIT_AS`)
- CPU 时间限制: 10s (`resource.RLIMIT_CPU`)
- 总超时: 30s (`subprocess.timeout`)
- 无网络访问

### 4.3 MoE 模型训练策略

Qwen3.5-35B-A3B 是 MoE 架构:
- **35B 总参数, ~3B 活跃参数** per token
- **256 experts**: 8 routed + 1 shared per layer
- **Hybrid Attention**: 30× Gated DeltaNet + 10× Full Attention
- **262K native context**

#### 问题: Router 参数敏感性

MoE 的 router (gate network) 决定每个 token 分配给哪些 experts。RL 训练中：
- Router 参数对 learning rate 极度敏感
- 少量 RL 数据可能导致路由分布剧烈震荡
- 破坏 pretrain 阶段学到的 expert 专业化

#### 解决方案

在训练脚本中传入冻结配置 (veRL 支持通过 FSDP 的 module-level freeze):

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.optim.lr=5e-6 \     # MoE 比 dense 更低
    # MoE router 冻结需要在模型代码中实现
    # 或通过 LoRA 只训练特定模块来间接实现
```

MoE LR 选择: 官方 veRL 示例中 Qwen2.5-7B (dense) 用 `5e-8`，我们的 MoE 模型用 `5e-6` (参考 Qwen3.5-SWE 的实践)。

#### Load Balancing 监控

训练时需要监控 load balancing loss：
- `load_balance_loss`: 应保持稳定
- `expert_utilization`: 各 expert 的利用率方差
- `router_entropy`: 路由决策的熵值

### 4.4 长上下文 (256K) 训练策略

#### 渐进式上下文扩展

直接在 256K 上下文训练有两个问题：(1) 显存不足，(2) 模型尚未学会处理长依赖。

分阶段扩展（通过在不同 epoch 调整 `data.max_response_length`）:

```
Phase    Step        Context   Batch   内存估计 (per GPU)
──────────────────────────────────────────────────────
A        0-500       32K       128     ~40GB
B        500-1000    64K       64      ~60GB
C        1000-1500   128K      32      ~80GB
D        1500+       256K      16      ~90GB (with SP)
```

#### 内存优化

| 技术 | veRL 参数 | 节省量 |
|-----|-----------|--------|
| Gradient Checkpointing | `actor_rollout_ref.model.enable_gradient_checkpointing=True` | ~40% activation |
| FSDP Param Offload | `actor_rollout_ref.actor.fsdp_config.param_offload=True` | ~30% model memory |
| FSDP Optimizer Offload | `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True` | ~50% optimizer |
| Ref Policy Offload | `actor_rollout_ref.ref.fsdp_config.param_offload=True` | ~30% ref model |
| vLLM Memory | `actor_rollout_ref.rollout.gpu_memory_utilization=0.4` | 控制推理显存 |

---

## 5. Qwen3.5-35B-A3B-Turbo-SWE 训练细节

参考实现的完整训练参数 ([HuggingFace](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1)):

| 阶段 | 参数 | 值 |
|------|------|-----|
| **数据** | 原始 sessions | 4,756 |
| | 提取 pairs | 4,551 |
| | 质量标注 (Claude Opus 4.6) | 3,580 pairs (avg reward 0.477) |
| **SFT** | 高质量 pairs (score>0.6) | 2,674 |
| | LoRA rank | 64 |
| | Precision | bf16 |
| | Steps | 670 |
| | Loss | 1.438 → 0.509 (-65%) |
| | Duration | ~3h |
| **GRPO Sampling** | Prompts | 200 |
| | Completions per prompt (G) | 8 |
| | Total completions | 1,600 |
| | Reward function | Execution-based (compile + run) |
| | Duration | ~2h |
| **GRPO Update** | Filter threshold | reward >= 0.5 |
| | Best completions | 161 (RFT) |
| | Steps | 123 |
| | Final loss | 1.97 |
| | Duration | ~12min |
| **硬件** | GPU | RTX PRO 6000 Blackwell (96GB) |
| | Total time | ~5.2h |

关键观察:
1. GRPO phase 中只保留 reward >= 0.5 的样本做 RFT (Rejection Fine-Tuning)
2. 总训练时间 ~5h 在单卡 96GB GPU 上
3. G=8 是较好的 group size 平衡点

---

## 6. 评估体系

### 6.1 训练期监控指标

| 指标 | veRL logging key | 健康范围 | 异常信号 |
|-----|-----------------|---------|---------|
| Actor loss | `actor/loss` | 持续下降 | 上升 = 不稳定 |
| KL loss | `actor/kl_loss` | < 0.1 | > 0.1 = 策略漂移 |
| Reward mean | `reward/mean` | 持续上升 | 停滞 = reward 问题 |
| Reward std | `reward/std` | 逐渐降低 | 持续高 = 没收敛 |
| Compile rate | custom | > 0.7 → > 0.85 | < 0.5 |
| Test pass rate | custom | > 0.3 → > 0.75 | 无提升 |

### 6.2 离线评估

| 指标 | 目标 | 方法 |
|-----|------|------|
| Compile Success (pass@1) | > 85% | 100 coding problems |
| Test Pass (pass@1) | > 75% | 单次生成 |
| Test Pass (pass@8) | > 90% | 8 次生成取最好 |
| Avg Reward | > 0.65 | execution-based |

---

## 7. 风险分析

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| Reward Hacking | 模型"骗"reward | 多元 reward 信号 + KL 约束 + 人工抽检 |
| 训练不稳定 | Loss 震荡 | 小 LR + 梯度裁剪 + 频繁 checkpoint |
| MoE Router 退化 | Expert 负载不均 | 冻结 router + 监控 load balancing |
| 显存 OOM | 训练中断 | 渐进式上下文 + FSDP offload |
| 灾难性遗忘 | 丢失 pretrain 能力 | LoRA (只更新 <5% 参数) |
| Reward 稀疏 | 梯度消失 | Dynamic sampling (DAPO) + reward shaping |

---

## 8. Roadmap

### Phase 1 (当前)
- 标准 GRPO pipeline (veRL `main_ppo` + `algorithm.adv_estimator=grpo`)
- 代码生成场景 (execution-based reward)
- Qwen3.5-35B-A3B (MoE)
- 32K-256K 渐进扩展

### Phase 2
- DAPO 算法替换 GRPO (长序列稳定性)
- 多轮工具调用训练 (agent trajectory reward)
- SWE-bench 评估
- veRL Reward Loop async 模式

### Phase 3
- Multi-agent 协作训练
- Online RL (持续从部署环境收集数据)
- 跨模型迁移 (把 RL 经验迁移到新 base model)
- 1M+ 上下文支持

---

## 9. 参考文献

1. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/pdf/2402.03300) — GRPO 原始论文
2. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO 在推理模型上的应用
3. [DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/DAPO.html) — GRPO 改进
4. [DrGRPO: Understanding R1-Zero-Like Training](https://arxiv.org/pdf/2503.20783) — GRPO length bias 分析
5. [KLong: Training LLM Agent for Extremely Long-horizon Tasks](https://arxiv.org/abs/2602.17547) — 长上下文 Agent RL
6. [Qwen3-Coder: Agentic Coding in the World](http://qwenlm.github.io/blog/qwen3-coder/) — 256K Agent RL 实践
7. [ATLAS: Scaling Agentic Capabilities](https://arxiv.org/abs/2603.06713) — 高效 Agent RL
8. [Qwen3.5-35B-A3B-Turbo-SWE](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) — 参考实现
9. [veRL Documentation](https://verl.readthedocs.io/) — 训练框架
10. [veRL Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html) — Reward 架构详解
11. [GRPO Series Algorithms (AReaL)](https://github.com/inclusionAI/AReaL/blob/main/docs/en/algorithms/grpo_series.md) — GRPO 家族算法汇总
