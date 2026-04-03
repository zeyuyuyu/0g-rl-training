# MoE 模型 RL 训练指南

## 1. MoE 架构回顾

### 1.1 什么是 MoE

Mixture-of-Experts (MoE) 模型将 Transformer 的 FFN 层替换为多个平行的 expert 网络，由 router (gate) 动态选择激活哪些 expert：

```
                Input Token
                    │
                    ▼
             ┌─────────────┐
             │   Router     │  (学习每个 token 分配给哪些 experts)
             │  (Gate Net)  │
             └──────┬──────┘
                    │ Top-K selection
           ┌────────┼────────┐
           ▼        ▼        ▼
       ┌───────┐ ┌───────┐ ┌───────┐
       │Expert1│ │Expert2│ │Expert3│ ... (256 experts)
       │ (FFN) │ │ (FFN) │ │ (FFN) │
       └───┬───┘ └───┬───┘ └───┬───┘
           │        │        │
           └────────┼────────┘
                    │ Weighted sum
                    ▼
              Output Token
```

### 1.2 Qwen3.5-35B-A3B 架构

| 参数 | 值 |
|-----|-----|
| 总参数量 | ~35B |
| 激活参数量 | ~3B per token |
| Expert 数量 | 256 |
| Routed experts (Top-K) | 8 |
| Shared experts | 1 |
| 注意力头数 | 40 |
| 隐藏维度 | 4096 |
| 层数 | 36 |

**关键特性**: 每个 token 只激活 8+1=9 个 expert，总 expert 有 256 个。这意味着每次前向传播只使用 ~3B 参数（35B 的 ~8.6%），推理效率极高。

## 2. MoE 训练的独特挑战

### 2.1 Router 参数敏感性

**问题**: MoE 的 router 决定了每个 token 分配给哪些 experts。Router 在 pretrain 期间学习了精细的 expert 专业化分工。RL 训练中如果更新 router 参数：

- **少量 RL 数据 << pretrain 数据**: router 在 RL 数据分布上过拟合
- **路由震荡**: 某些 token 在不同 expert 之间来回跳转
- **Expert 退化**: 本来专门处理数学的 expert 被 RL 数据"污染"

**实验观察** (Qwen3.5 SWE 训练):
```
┌─────────────────────────────────────────┐
│ 更新 Router 的后果:                       │
│                                          │
│ Step 0:    Expert 0→数学, 1→代码, 2→语言  │
│ Step 100:  Expert 0→混合, 1→代码, 2→代码  │
│ Step 200:  所有 Expert 都在处理代码        │
│                                          │
│ → Expert 多样性丧失 → 非代码能力下降       │
└─────────────────────────────────────────┘
```

**最佳实践**: 完全冻结 router 参数。

### 2.2 Load Balancing

**问题**: MoE 模型需要 experts 之间的负载均衡。如果某些 expert 被过度使用，会导致：
- GPU 利用率不均
- 被过度使用的 expert 成为瓶颈
- 训练速度下降

**监控指标**:

```python
# 在 veRL callback 中监控
def log_moe_metrics(model):
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'router'):
            routing_weights = layer.router.last_routing_weights
            expert_load = routing_weights.sum(dim=0)

            metrics = {
                f"layer_{layer_idx}/load_std": expert_load.std().item(),
                f"layer_{layer_idx}/load_max": expert_load.max().item(),
                f"layer_{layer_idx}/load_min": expert_load.min().item(),
                f"layer_{layer_idx}/router_entropy": -(routing_weights * routing_weights.log()).sum().item(),
            }
```

**健康指标**:
| 指标 | 健康范围 | 异常信号 |
|-----|---------|---------|
| load_std / load_mean | < 0.3 | > 0.5 = 负载不均 |
| router_entropy | 接近 log(K) | 远低于 log(K) = 路由退化 |
| max_load / min_load | < 3 | > 5 = 严重不均 |

## 3. 训练配置

### 3.1 冻结策略

veRL 本身没有内置 `freeze_modules` 配置，需要通过以下方式实现:

1. **LoRA 策略**: 只对 attention + expert FFN 添加 LoRA adapter，router 自然不会被更新
2. **模型代码级冻结**: 在模型加载后手动冻结 router 参数

```python
# 在模型加载后冻结 router
for name, param in model.named_parameters():
    if any(kw in name for kw in ("router", "gate", "shared_expert_gate")):
        param.requires_grad = False
```

**可训练参数**: Attention (Q/K/V/O)、Expert FFN (up/gate/down_proj)、LayerNorm

**不可训练参数**: Router、Gate、Embedding、LM Head (由 LoRA 配置控制)

### 3.2 学习率

```yaml
optim:
  lr: 5e-6                    # Dense 模型用 1e-5
  lr_scheduler: cosine
  warmup_ratio: 0.03          # 比 dense 更长的 warmup
  weight_decay: 0.01
```

**为什么更小的 LR?**

MoE 的 expert FFN 参数在 pretrain 时已经高度专业化。大的 LR 会导致 expert 参数跳出当前 basin，丢失专业化信息。经验规则：MoE 的 LR 是 Dense 的 0.3-0.5 倍。

### 3.3 LoRA 配置

```yaml
lora:
  r: 64
  alpha: 128
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj          # Expert FFN gate
    - up_proj            # Expert FFN up
    - down_proj          # Expert FFN down
  modules_to_save: []    # 不额外保存任何 module
```

**注意**: `gate_proj` 这里是 Expert 内部的 FFN gate (SwiGLU)，不是 Router gate。命名容易混淆。

### 3.4 梯度裁剪

```yaml
optim:
  max_grad_norm: 0.5         # 比 dense (1.0) 更激进的裁剪
```

MoE 模型的梯度更容易出现尖峰（因为只有部分 expert 收到梯度），需要更强的裁剪。

## 4. 显存优化

### 4.1 MoE 的显存特点

MoE 模型的参数量大（35B）但激活参数少（3B）。这意味着：

- **前向**: 只需要激活 3B 参数，显存友好
- **存储**: 需要存储全部 35B 参数，显存大
- **LoRA 优势**: 只在被激活的 expert 上添加 LoRA，存储开销小

```
                35B MoE 模型显存构成
┌──────────────────────────────────────────────┐
│ Model Weights          ~70 GB (bf16)         │
│ LoRA Adapters          ~0.5 GB (仅 active)   │
│ Optimizer States       ~1.0 GB (LoRA only)   │
│ Activations (32K ctx)  ~10 GB                │
│ Activations (256K ctx) ~80 GB                │
│ KV Cache (rollout)     ~5 GB (32K)           │
│──────────────────────────────────────────────│
│ Total (32K):           ~87 GB per GPU        │
│ Total (256K):          ~157 GB → 需要多卡    │
└──────────────────────────────────────────────┘
```

### 4.2 多卡策略

| GPU 数量 | 上下文长度 | 策略 |
|---------|-----------|------|
| 1× A100 80GB | 32K | LoRA + activation checkpointing |
| 2× A100 80GB | 64K | FSDP (模型 shard) |
| 4× A100 80GB | 128K | FSDP + sequence parallel |
| 8× H100 80GB | 256K | FSDP + ring attention |

## 5. 常见问题与解决

### Q: 训练 loss 不降，reward 不涨

**可能原因**:
1. Router 被错误更新（检查 `freeze_modules`）
2. LR 太大导致 expert 震荡（降到 1e-6）
3. 梯度裁剪太激进（检查 `grad_norm` 指标是否一直被 clip）

**调试步骤**:
```bash
# 1. 检查哪些参数在更新
python -c "
import torch
ckpt = torch.load('checkpoint.pt')
for name, param in ckpt['model'].items():
    if 'router' in name or 'gate' in name:
        print(f'{name}: grad={param.requires_grad}')
"

# 2. 对比两个 checkpoint 的 router 参数
# 如果 router 参数变化了，说明冻结没生效
```

### Q: Expert 利用率不均

**症状**: 某些 expert 处理了 80% 的 token，大部分 expert 闲置。

**解决方案**:
1. 确认 pretrain 模型的 load balancing 正常（是否在训练中退化）
2. 添加 auxiliary load balancing loss（如果 veRL 支持）
3. 增加 batch size，让 router 看到更多样的 token

### Q: LoRA merge 后性能下降

**可能原因**: LoRA 的 alpha/r 比例不当导致 merge 后参数 scale 变化。

**解决方案**:
```python
# merge 后检查参数 scale
model = merge_lora(base_model, lora_adapter)
for name, param in model.named_parameters():
    if 'expert' in name:
        print(f'{name}: mean={param.mean():.4f}, std={param.std():.4f}')
# 与 base_model 对比，确保 scale 没有量级变化
```

## 6. 与 Dense 模型训练的对比

| 维度 | Dense 模型 | MoE 模型 |
|-----|-----------|---------|
| 学习率 | 1e-5 | 5e-6 (0.5×) |
| Warmup | 3% | 3-5% |
| 梯度裁剪 | 1.0 | 0.5 |
| 冻结策略 | 无特殊 | 冻结 router/gate |
| 显存 | 与参数量线性 | 存储大，计算小 |
| LoRA target | attention + FFN | attention + expert FFN |
| Load balancing | 不需要 | 需要监控 |
| 推理速度 | 与参数量线性 | 仅 active 参数相关 |

## 7. 参考资料

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - MoE 架构细节
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) - HuggingFace MoE 教程
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - MoE 路由机制
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - MoE + RL 实践
