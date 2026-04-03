# RL Training for 0G Platform

基于 [veRL](https://verl.readthedocs.io/) 框架的 RL 训练代码，支持 GRPO (Group Relative Policy Optimization) 算法，参考 [Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) 的训练流程。

## 目录结构

```
rl-training/
├── configs/
│   └── grpo_qwen.yaml          # GRPO 训练配置
├── scripts/
│   ├── train_grpo.sh          # 训练启动脚本
│   └── export_for_0g.py       # 导出模型到 0G 平台
├── src/
│   ├── data_processor.py      # 数据处理
│   ├── reward_functions.py    # Reward 函数实现
│   └── evaluate.py            # 评估工具
├── data/                      # 训练数据
├── checkpoints/               # 模型检查点
├── logs/                      # 训练日志
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n verl python=3.10
conda activate verl

# 安装依赖
pip install -r requirements.txt

# 安装 veRL (如果不在 requirements 中)
pip install verl

# 安装 vLLM (用于 rollout generation)
pip install vllm==0.8.2

# Install pycryptodome for accurate keccak256 (optional but recommended)
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

# 或者从 SFT 数据转换
python src/data_processor.py \
    --input ./raw/sft_data.jsonl \
    --output-dir ./data \
    --format sft \
    --val-ratio 0.1
```

### 3. 启动训练

```bash
# 单卡训练
./scripts/train_grpo.sh single

# 单机多卡 (4 GPUs)
./scripts/train_grpo.sh multi 4

# SLURM 集群
./scripts/train_grpo.sh slurm

# 导出到 0G 平台
./scripts/train_grpo.sh 0g
```

或者直接使用 Python:

```bash
# 单机多卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    -m verl.trainer.main_ppo \
    config_path=./configs/grpo_qwen.yaml \
    algorithm.adv_estimator=grpo \
    actor_rollout.actor.use_kl_loss=True
```

### 4. 导出到 0G 平台

```bash
# 导出模型并计算 hash
python scripts/export_for_0g.py \
    --checkpoint ./checkpoints/grpo/final \
    --output ./models/0g_ready \
    --compute-hash \
    --generate-config
```

## 训练流程详解

参考 [Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) 的完整 pipeline：

```
Step 1: 数据准备 (4,551 pairs from 4,756 sessions)
   ↓
Step 2: 质量标注 (Claude Opus 4.6, avg reward 0.477)
   ↓
Step 3: SFT 预训练 (bf16 LoRA r=64, 2,674 high-quality pairs)
   ↓
Step 4: GRPO RL 训练 ← 本项目重点
   │
   ├─ Group Sampling: G=8 (每个 prompt 生成 8 个 completion)
   ├─ Reward: Execution-based (compile + test + style)
   ├─ No Critic Model (vs PPO 省 40-60% 显存)
   └─ KL Regularization: add to loss, not reward
   ↓
Step 5: 模型导出 (LoRA merge + 0G hash 计算)
   ↓
Step 6: 部署到 0G 网络
```

## 关键配置说明

### GRPO 特有配置

```yaml
algorithm:
  adv_estimator: grpo              # 使用 GRPO 而不是 GAE/PPO
  norm_adv_by_std_in_grpo: true   # 使用 group std 归一化

actor_rollout:
  ref:
    rollout:
      n: 8                         # Group size，每个 prompt 采样 8 个
  actor:
    use_kl_loss: true              # GRPO 推荐：KL 加到 loss 而不是 reward
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl       # KL 计算方式
    loss_agg_mode: token-mean      # loss 聚合方式（推荐，长 CoT 稳定）
```

### MoE 模型注意事项

```yaml
actor_rollout:
  actor:
    optim:
      lr: 5e-6                     # MoE 用更小的 LR (dense 用 1e-5)
    freeze_modules:                # 冻结 router 防止 expert 震荡
      - router
      - gate
      - shared_expert_gate
```

### 长上下文 (256K) 配置

```yaml
long_context:
  context_schedule:                # 渐进式扩展上下文长度
    - [0, 32768]                   # Step 0-500: 32K
    - [500, 65536]                 # Step 500-1000: 64K
    - [1000, 131072]               # Step 1000-1500: 128K
    - [1500, 262144]               # Step 1500+: 256K
```

## Reward 函数设计

基于 [execution-based reward](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) 的实现：

```python
# reward_functions.py

class CodeExecutionReward:
    def compute_reward(self, completion: str) -> float:
        # 1. 提取代码块 (30%)
        code = self.extract_code(completion)
        
        # 2. 语法/编译检查 (30%)
        syntax_ok = self.check_syntax(code)
        
        # 3. 测试执行 (50%)
        test_passed = self.run_tests(code)
        
        # 4. 代码风格 (20%)
        style_score = self.check_style(code)
        
        return 0.3 * compile_score + 0.5 * test_score + 0.2 * style_score
```

## 监控与评估

### 训练监控

```bash
# TensorBoard
tensorboard --logdir ./logs

# Weights & Biases (在 config 中配置)
trainer.logging:
  project_name: 0g-rl-training
  experiment_name: qwen-grpo-256k
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

评估指标：
- **Avg Reward**: 平均 reward (目标: >0.6)
- **Compile Rate**: 代码编译成功率 (目标: >85%)
- **Test Pass Rate**: 测试通过率 (目标: >75%)
- **Long Context Retention**: 长上下文信息保持率

## 与 0G 平台集成

### 1. 添加模型到 SCRIPT_MAP

```go
// api/fine-tuning/const/const.go
SCRIPT_MAP = map[string]ModelConfig{
    // 添加你的模型
    "0x[你的模型hash]": {
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

### 3. 部署到 CVM

```bash
# 复制模型到 CVM
phala ssh <cvm-id> -- "mkdir -p /dstack/persistent/models/your-rl-model"
docker save your-model | gzip > /tmp/model.tar.gz
# 上传并解压...
```

## 预期效果

参考 [Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1) 的效果：

| 指标 | SFT 后 | GRPO 后 | 提升 |
|-----|--------|---------|------|
| Compile Success | ~60% | ~85% | +25% |
| Test Pass | ~40% | ~75% | +35% |
| Avg Reward | 0.35 | 0.65 | +86% |

## 常见问题

### Q: GRPO 和 PPO 的区别？

A: GRPO 不需要 critic 模型，直接用 group baseline 计算 advantage。省 40-60% 显存，训练成本约为 1/18。

### Q: MoE 模型 RL 训练要注意什么？

A: 冻结 router 参数，使用更小的 learning rate (5e-6 vs 1e-5)，监控 load balancing loss。

### Q: 长上下文 (256K) 怎么训练？

A: 渐进式扩展：32K → 64K → 128K → 256K。使用 activation checkpointing 和 sequence parallelism。

### Q: Reward hacking 怎么解决？

A: 多样化 reward 信号（compile + test + style），定期人工抽检，使用 KL regularization。

## 参考资源

- [veRL Documentation](https://verl.readthedocs.io/)
- [GRPO Algorithm](https://verl.readthedocs.io/en/v0.5.x/algo/grpo.html)
- [Qwen3.5-35B-A3B-Turbo-SWE](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1)
- [DeepSeekMath Paper](https://arxiv.org/pdf/2402.03300) (GRPO 原始论文)

## 开发团队

- RL Training Pipeline: Zeyu
- Reward Function Design: Mart
- 0G Platform Integration: William

## License

Apache 2.0
