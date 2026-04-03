# RL Training for 0G Platform

基于 [veRL](https://verl.readthedocs.io/) 框架的 RL 训练代码，采用 **GRPO (Group Relative Policy Optimization)** 算法，专为 0G Compute Network 的长上下文（256K）Agentic AI 模型设计。

> **参考实现**: [Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) 的 SFT+GRPO 训练流程

---

## 🎯 设计目标

```
┌─────────────────────────────────────────────────────────────────┐
│                        核心挑战                                  │
├─────────────────────────────────────────────────────────────────┤
│ 1. 长上下文 (256K) → 渐进式训练策略                              │
│ 2. MoE 模型稳定性 → Router 冻结 + 低 LR                         │
│ 3. Agentic 能力提升 → Execution-based Reward                     │
│ 4. 0G 平台集成 → 自动 Hash 计算 + 配置生成                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RL Training Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐ │
│  │   Data Layer    │        │  Training Core  │        │   0G Export     │ │
│  │                 │───────▶│                 │───────▶│                 │ │
│  │ • Session Extraction   │ │ • GRPO Algorithm│        │ • LoRA Merge    │ │
│  │ • Quality Scoring      │ │ • vLLM Rollout  │        │ • Keccak256     │ │
│  │ • Parquet Format       │ │ • Reward Compute│        │ • Config Gen    │ │
│  └─────────────────┘        └─────────────────┘        └─────────────────┘ │
│           │                         │                          │             │
│           ▼                         ▼                          ▼             │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐   │
│  │ Input:          │        │ Key Params:     │        │ Output:         │   │
│  │ - 4,551 pairs   │        │ - Group Size: 8 │        │ - Model Hash    │   │
│  │ - Quality >0.4  │        │ - LR: 5e-6      │        │ - Price Coeff   │   │
│  │ - Test Cases    │        │ - LoRA r=64     │        │ - Storage Fee   │   │
│  └─────────────────┘        └─────────────────┘        └─────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 完整训练流程

### Phase 1: 数据准备 (Data Preparation)

```
┌────────────────────────────────────────────────────────────────────┐
│  Raw Agent Sessions  →  Quality Filter  →  Train/Val Split          │
│                                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │ 4,756       │     │ 3,580       │     │ Train: 3,222        │   │
│  │ sessions    │────▶│ pairs       │────▶│ (90%)               │   │
│  │ (Codex/     │     │ (score>0.4) │     │                     │   │
│  │  Claude/    │     │             │     │ Val: 358            │   │
│  │  OpenCode)   │     │             │     │ (10%)               │   │
│  └─────────────┘     └─────────────┘     └─────────────────────┘   │
│                                                                    │
│  Quality Scoring:                                                  │
│  • Compile Success (30%)                                           │
│  • Test Pass Rate (50%)                                            │
│  • Code Style (20%)                                                │
└────────────────────────────────────────────────────────────────────┘
```

**关键设计**: 使用 execution-based quality score（而不是人工打分），确保数据质量可客观验证。

```bash
# 数据准备命令
python src/data_processor.py \
    --input ./raw/sessions.jsonl \
    --output-dir ./data \
    --format opencode \
    --min-quality 0.4
```

---

### Phase 2: SFT 预热

```
┌────────────────────────────────────────────────────────────────────┐
│  SFT Phase: Supervised Fine-Tuning on High-Quality Data              │
│                                                                    │
│  Dataset: 2,674 high-quality pairs (score > 0.6)                    │
│  Config: bf16, LoRA r=64, 670 steps                                │
│  Loss: 1.438 → 0.509 (-65% improvement)                            │
│                                                                    │
│  Purpose:                                                          │
│  • 建立基础 coding capability                                     │
│  • 学习正确的 code format/style                                   │
│  • 为 RL 阶段提供好的 init checkpoint                             │
└────────────────────────────────────────────────────────────────────┘
```

---

### Phase 3: GRPO RL 训练

这是整个项目的核心。我们选择 **GRPO** 而不是传统 PPO，基于以下关键考虑：

#### Why GRPO?

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO vs PPO Comparison                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PPO (Traditional)              GRPO (Our Choice)               │
│   ─────────────────              ───────────────               │
│                                                                 │
│   ┌───────────────┐              ┌───────────────┐             │
│   │ Actor Model   │              │ Actor Model   │             │
│   │  (Policy)     │              │  (Policy)     │             │
│   └───────┬───────┘              └───────┬───────┘             │
│           │                              │                     │
│   ┌───────▼───────┐                      │                     │
│   │ Critic Model  │                      │                     │
│   │ (Value Func)  │                      │                     │
│   └───────┬───────┘                      │                     │
│           │                              │                     │
│   ┌───────▼───────┐              ┌───────▼───────┐             │
│   │ Advantage =  │              │ Group Baseline│             │
│   │ R - V(s)     │              │ (No Critic)   │             │
│   └───────────────┘              └───────────────┘             │
│                                                                 │
│   Memory: 2× Actor                 Memory: 1.2× Actor          │
│   Cost: Baseline                   Cost: ~1/18 of PPO          │
│   Stability: Depends on V(s)     Stability: More stable       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### GRPO 算法详解

```python
# 对于每个 prompt:
# 1. 生成 G=8 个 completions (Group Sampling)
completions = [c1, c2, c3, c4, c5, c6, c7, c8]

# 2. 计算每个 completion 的 reward
rewards = [r1, r2, r3, r4, r5, r6, r7, r8]

# 3. 计算 group baseline
baseline = mean(rewards)
std = std(rewards)

# 4. 计算 advantage (relative to group)
advantages = [(ri - baseline) / std for ri in rewards]

# 5. Policy update (clipped, with KL penalty)
# 优于 group average 的 completion 被鼓励
# 劣于 group average 的 completion 被抑制
```

#### 训练配置

```yaml
# GRPO 核心参数
algorithm:
  adv_estimator: grpo              # 使用 GRPO 而不是 GAE
  norm_adv_by_std_in_grpo: true   # Group 内 std 归一化

actor_rollout:
  ref:
    rollout:
      n: 8                         # Group Size = 8
  actor:
    use_kl_loss: true              # KL 加到 loss (GRPO 推荐)
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl       # 低方差 KL 估计
    loss_agg_mode: token-mean      # Token-level loss (长 CoT 稳定)
```

#### MoE 模型特殊处理

对于 Qwen3.5-35B-A3B (MoE) 模型，我们采取以下措施防止 expert 崩溃：

```yaml
actor_rollout:
  actor:
    optim:
      lr: 5e-6                     # MoE 用更小的 LR (dense 用 1e-5)
    
    # 冻结 router 参数，防止 SFT 阶段学到的 expert 专业化被破坏
    freeze_modules:
      - router
      - gate
      - shared_expert_gate
```

---

### Phase 4: 长上下文渐进式扩展 (256K)

```
┌────────────────────────────────────────────────────────────────────┐
│           Context Length Schedule (Progressive Expansion)           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Step 0-500:      32K  tokens                                     │
│       ↓                                                           │
│  Step 500-1000:   64K  tokens                                     │
│       ↓                                                           │
│  Step 1000-1500:  128K tokens                                     │
│       ↓                                                           │
│  Step 1500+:      256K tokens (Target)                              │
│                                                                    │
│  Why Progressive?                                                  │
│  • 避免显存爆炸 (OOM)                                             │
│  • 让模型逐步适应长依赖                                           │
│  • 与 activation checkpointing 配合                               │
│                                                                    │
│  Memory Optimizations:                                            │
│  • enable_activation_checkpointing: true                         │
│  • use_ring_attention: false (set true for >128K)                │
│  • sequence_parallel_size: 1 (increase for very long contexts)   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🎁 Reward Function 设计 (核心创新)

我们采用 **Execution-based Reward**，这是防止 reward hacking 的关键设计。

### 为什么不用 LLM-as-Judge?

```
┌─────────────────────────────────────────────────────────────────┐
│               LLM-as-Judge vs Execution-based                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM-as-Judge (传统方法)          Execution-based (我们的方法)  │
│  ──────────────────────          ─────────────────────────      │
│                                                                 │
│  Prompt: "Rate this code 1-10"    实际运行代码                  │
│           ↓                       ┌──────────────┐              │
│  LLM Output: "8/10"               │ Python Exec │              │
│           ↓                       │  - Parse    │              │
│  Reward: 0.8                      │  - Compile  │              │
│                                   │  - Test     │              │
│  问题:                              └──────────────┘              │
│  • LLM 可能 favor 某种 pattern      结果:                        │
│  • 难以捕捉 runtime error          • 客观可验证                  │
│  • 容易被 exploit                  • 无法被 hack                 │
│                                    • 真实性能反映                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Reward 组成

```python
class CodeExecutionReward:
    """
    Reward = 0.3 × Compile + 0.5 × Test + 0.2 × Style
    """
    
    def compute_reward(self, completion: str) -> float:
        # 1. 提取代码 (如果失败，reward=0)
        code = extract_code(completion)
        if not code:
            return 0.0
        
        # 2. 编译/语法检查 (30%)
        # 使用 Python AST parser
        compile_score = 1.0 if ast.parse(code) else 0.0
        
        # 3. 测试执行 (50%)
        # 在沙箱环境中运行，检查 test cases
        test_score = run_tests(code) / total_tests
        
        # 4. 代码风格 (20%)
        # Docstring, line length, imports position
        style_score = check_style(code)
        
        return 0.3 * compile_score + 0.5 * test_score + 0.2 * style_score
```

### Test Cases 来源

- 从训练数据中提取（用户提供的测试用例）
- 自动生成（针对简单函数）
- 编译测试（对于复杂项目）

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n verl python=3.10
conda activate verl

# 安装依赖
pip install -r requirements.txt

# 安装 veRL
pip install verl

# 安装 vLLM (用于 fast rollout generation)
pip install vllm==0.8.2

# 安装 keccak256 (用于准确的模型 hash)
pip install pycryptodome
```

### 2. 数据准备

```bash
# 从 OpenCode/Claude Code 会话提取数据
python src/data_processor.py \
    --input ./raw/opencode_sessions.jsonl \
    --output-dir ./data \
    --format opencode \
    --min-quality 0.4

# 生成的数据格式 (Parquet)
# - prompt: str
# - response: str (reference)
# - data_source: str
# - prompt_metadata: JSON string (包含 test_cases)
```

### 3. 启动训练

```bash
# 本地测试 (小模型，快速验证)
./scripts/train_grpo.sh single

# 单机多卡 (推荐)
./scripts/train_grpo.sh multi 4

# SLURM 集群
./scripts/train_grpo.sh slurm

# 0G 部署模式 (自动导出)
./scripts/train_grpo.sh 0g
```

### 4. 导出到 0G 平台

```bash
python scripts/export_for_0g.py \
    --checkpoint ./checkpoints/grpo/final \
    --output ./models/0g_ready \
    --compute-hash \
    --generate-config

# 输出:
# - 模型文件 (LoRA merged)
# - model_hash.txt (keccak256)
# - 0g_config.json (SCRIPT_MAP entry)
```

---

## 📈 预期效果

参考 [Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) 的训练结果：

| 阶段 | Loss | 时间 | 硬件 |
|-----|------|------|------|
| SFT | 1.438 → 0.509 | ~3h (670 steps) | RTX PRO 6000 96GB |
| GRPO Sampling | - | ~2h (1,600 completions) | Same |
| GRPO Update | 1.97 | ~12min (123 steps) | Same |

**性能提升预期**:

| 指标 | SFT 后 | GRPO 后 | 提升 |
|-----|--------|---------|------|
| Compile Success | ~60% | ~85% | +25% |
| Test Pass | ~40% | ~75% | +35% |
| Avg Reward | 0.35 | 0.65 | +86% |

---

## 🔧 0G 平台集成

### 1. 添加模型到 SCRIPT_MAP

```go
// api/fine-tuning/const/const.go
SCRIPT_MAP = map[string]ModelConfig{
    // 你的 RL 训练模型
    "0x[model_hash_from_export]": {
        TrainingScript:   "/app/train_lora.py",
        PriceCoefficient: 4,  // 7B 模型级别
        StorageFee:       15000000000000000, // ~150MB LoRA
    },
}
```

### 2. 配置 user_config.yaml

```yaml
ModelLocalPaths:
  "0x[hash]": "/dstack/persistent/models/your-rl-model"

ModelHuggingFaceFallback:
  "0x[hash]": "your-hf-org/your-model"
```

### 3. 部署

```bash
# 1. 复制模型到 CVM
phala ssh <cvm-id> -- "mkdir -p /dstack/persistent/models/your-rl-model"

# 2. 打包并上传
docker save your-model | gzip > /tmp/model.tar.gz
# ... 上传过程 ...

# 3. 重启 fine-tuning broker
```

---

## 🛠️ 监控与调试

### 训练监控

```bash
# TensorBoard
tensorboard --logdir ./logs

# 关键指标:
# - actor/loss: 应逐渐下降
# - actor/kl_loss: 监控不要暴涨 (>0.1 需要调整)
# - reward/mean: 应逐渐上升
# - compile_rate: 目标 >85%
# - test_pass_rate: 目标 >75%
```

### 模型评估

```bash
# 单模型评估
python src/evaluate.py \
    --model ./checkpoints/grpo/final \
    --test-data ./data/val.jsonl \
    --use-vllm \
    --output ./eval_results

# 对比 RL vs SFT
python src/evaluate.py \
    --model ./checkpoints/grpo/final \
    --test-data ./data/val.jsonl \
    --compare-with ./checkpoints/sft/final \
    --output ./comparison
```

---

## 📚 核心设计决策总结

| 决策 | 选择 | 理由 |
|-----|------|------|
| RL 算法 | GRPO | 无 critic，省显存，成本低，稳定 |
| Reward | Execution-based | 客观可验证，防 reward hacking |
| 训练策略 | 渐进式扩展 | 32K→256K，避免 OOM，逐步适应 |
| MoE 优化 | 冻结 router | 防止 expert 崩溃 |
| 导出格式 | LoRA merge | 0G 平台兼容，大小适中 |

---

## 🔗 参考资源

- [veRL Documentation](https://verl.readthedocs.io/)
- [GRPO Algorithm](https://verl.readthedocs.io/en/v0.5.x/algo/grpo.html)
- [Qwen3.5-35B-A3B-Turbo-SWE](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1)
- [DeepSeekMath Paper](https://arxiv.org/pdf/2402.03300)

---

## 👥 开发团队

- **RL Training Pipeline**: Zeyu
- **Reward Function Design**: Mart
- **0G Platform Integration**: William

## License

Apache 2.0
