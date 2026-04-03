# RL Training Design Document

## 概述

本项目为 0G Compute Network 提供基于 GRPO (Group Relative Policy Optimization) 的 RL 训练能力，使 fine-tuned 模型具备更强的 agentic 能力。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      RL Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │ Data Sources │      │  SFT Model   │      │  RL Training │ │
│  │              │      │              │      │              │ │
│  │ • Codex      │─────▶│  (Checkpoint)│─────▶│   GRPO       │ │
│  │ • Claude     │      │              │      │   Algorithm  │ │
│  │ • OpenCode   │      └──────────────┘      └──────┬───────┘ │
│  │ • SFT Data   │                                   │         │
│  └──────────────┘                                   │         │
│                                                     ▼         │
│                                          ┌─────────────────┐  │
│                                          │ Reward Function │  │
│                                          │                 │  │
│                                          │ • Compile       │  │
│                                          │ • Test          │  │
│                                          │ • Style         │  │
│                                          └─────────────────┘  │
│                                                     │         │
│                                                     ▼         │
│                                          ┌─────────────────┐  │
│                                          │  0G Export      │  │
│                                          │                 │  │
│                                          │ • LoRA Merge    │  │
│                                          │ • Hash Compute  │  │
│                                          │ • Config Gen    │  │
│                                          └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 技术选型

| 组件 | 选择 | 理由 |
|-----|------|------|
| RL Framework | [veRL](https://verl.readthedocs.io/) | 生产级，支持 GRPO，集成 vLLM |
| Algorithm | GRPO | 无 critic 模型，省显存，成本低 |
| Inference | vLLM | 快速 rollout generation |
| Training | LoRA | 参数高效，适合 256K 上下文 |
| Reward | Execution-based | 客观可验证，防 reward hacking |

## 数据流

```
Raw Sessions → Extraction → Quality Filter → Parquet → veRL Dataset
     │              │              │              │           │
     ▼              ▼              ▼              ▼           ▼
 JSON/JSONL    Code+Tests    Score>0.4     veRL format    Train Batch
  4,756         4,551         3,580           ...           128 prompts
                                                    ↓
                                            128 × 8 = 1,024 completions
```

## Reward Function 设计

### 代码场景

```python
Reward = 0.3 × Compile + 0.5 × Test + 0.2 × Style
```

| Component | 权重 | 计算方式 |
|-----------|------|---------|
| Compile | 30% | Python AST parse success |
| Test | 50% | Unit test pass rate |
| Style | 20% | Docstring, line length, imports |

### 通用 Agent 场景 (可选)

```python
Reward = 0.5 × ToolSuccess + 0.3 × TaskCompletion + 0.2 × Efficiency
```

## GRPO vs PPO 对比

| 特性 | PPO | GRPO (本项目) |
|-----|-----|---------------|
| Critic Model | 需要独立训练 | ❌ 不需要 |
| 显存占用 | 2× actor | ~1.2× actor |
| 训练成本 | 基准 | 约 1/18 |
| Advantage | Value baseline | Group relative baseline |
| 稳定性 | 依赖 critic 质量 | 更稳定 |

## MoE 模型特殊处理

```yaml
# 冻结 router 防止 expert 震荡
freeze_modules:
  - router
  - gate
  - shared_expert_gate

# 更小的学习率
optim:
  lr: 5e-6  # (dense 模型用 1e-5)
```

## 长上下文 (256K) 策略

### 渐进式扩展

```
Step 0-500:    32K  context
Step 500-1000: 64K  context
Step 1000-1500: 128K context
Step 1500+:     256K context
```

### 内存优化

- Activation Checkpointing
- Flash Attention 2
- Sequence Parallelism (当扩展到 256K)

## 与 0G 平台集成

### 训练后导出

```bash
python scripts/export_for_0g.py \
    --checkpoint ./checkpoints/grpo/final \
    --output ./models/0g_ready \
    --compute-hash
```

### 生成配置

```json
{
  "model_hash": "0x...",
  "price_coefficient": 4,
  "storage_fee": "15000000000000000",
  "const_go_entry": "...",
  "user_config_entry": "..."
}
```

### 部署流程

1. 计算 model hash
2. 更新 `api/fine-tuning/const/const.go`
3. 更新 `user_config.yaml`
4. 上传模型到 CVM
5. 重启 fine-tuning broker

## 性能预期

参考 [Qwen3.5-35B-A3B-Turbo-SWE](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1):

| 阶段 | Loss | 时间 | 硬件 |
|-----|------|------|------|
| SFT | 1.438 → 0.509 | ~3h (670 steps) | RTX PRO 6000 96GB |
| GRPO Sampling | - | ~2h (1,600 completions) | Same |
| GRPO Update | 1.97 | ~12min (123 steps) | Same |

## 监控指标

### 训练期

- `actor/loss`: 策略 loss (应下降)
- `actor/kl_loss`: KL divergence (监控不暴涨)
- `reward/mean`: 平均 reward (应上升)

### 评估期

- Compile Success Rate: 目标 >85%
- Test Pass Rate: 目标 >75%
- Long Context Accuracy: 按长度分层测试

## 风险与缓解

| 风险 | 缓解措施 |
|-----|---------|
| Reward Hacking | 多样化 reward 信号 + KL regularization |
| 训练不稳定 | 小 learning rate + 梯度裁剪 |
| MoE Router 崩 | 冻结 router 参数 |
| 显存爆炸 | 渐进式上下文扩展 + activation checkpointing |
| 灾难性遗忘 | LoRA (只更新少量参数) |

## 扩展计划

### Phase 1 (当前)
- 代码生成场景
- 单 agent
- 256K 上下文

### Phase 2
- Multi-agent 协作
- 工具调用优化
- 1M 上下文

### Phase 3
- 在线学习
- 人类反馈集成
- 跨模型迁移

## 参考资源

- [veRL GitHub](https://github.com/volcengine/verl)
- [GRPO Paper](https://arxiv.org/pdf/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Qwen3.5 SWE Model](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1)
