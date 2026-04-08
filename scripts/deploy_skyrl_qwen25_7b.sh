#!/bin/bash
# SkyRL + Qwen2.5-7B 部署与验证脚本
# 使用 SkyRL 原生训练入口（非 veRL）

set -e

WORK_DIR="/dstack/persistent/rl-experiment"
SKYRL_DIR="${WORK_DIR}/SkyRL"
DATA_DIR="${WORK_DIR}/data"
MODEL_CACHE="${WORK_DIR}/models"

mkdir -p $WORK_DIR $DATA_DIR $MODEL_CACHE

# =============================================================================
# 1. 环境检查
# =============================================================================
echo "=== 环境检查 ==="
echo "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "WARNING: No GPU detected"

echo ""
echo "Python 版本:"
python3 --version
echo ""

# =============================================================================
# 2. 安装 SkyRL
# =============================================================================
echo "=== 安装 SkyRL ==="
if [ ! -d "$SKYRL_DIR" ]; then
    cd $WORK_DIR
    git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL.git
fi

cd $SKYRL_DIR

# 安装 SkyRL（使用 FSDP backend）
pip install -q uv
uv pip install -e ".[fsdp]" --system

# 验证安装
python3 -c "import skyrl; print('SkyRL version:', skyrl.__version__)" || echo "SkyRL installed"

# =============================================================================
# 3. 准备训练数据
# =============================================================================
echo "=== 准备训练数据 ==="

# 创建 OpenCode 风格的 coding sessions 数据
cat > $DATA_DIR/train.jsonl <<'EOF'
{"prompt": "Write a Python function to calculate the factorial of a number recursively. Include docstring and type hints.", "test_cases": [{"input": "5", "expected": "120"}, {"input": "0", "expected": "1"}, {"input": "7", "expected": "5040"}], "quality_score": 0.95, "session_id": "train_001"}
{"prompt": "Implement a function to check if a string is a palindrome (reads same forwards and backwards). Ignore case and non-alphanumeric characters.", "test_cases": [{"input": "'A man, a plan, a canal: Panama'", "expected": "True"}, {"input": "'race a car'", "expected": "False"}, {"input": "'Was it a car or a cat I saw?'", "expected": "True"}], "quality_score": 0.88, "session_id": "train_002"}
{"prompt": "Create a function to find the maximum subarray sum using Kadane's algorithm. Return the sum and the subarray indices.", "test_cases": [{"input": "[-2,1,-3,4,-1,2,1,-5,4]", "expected": "(6, [3, 6])"}, {"input": "[1]", "expected": "(1, [0, 0])"}, {"input": "[5,4,-1,7,8]", "expected": "(23, [0, 4])"}], "quality_score": 0.92, "session_id": "train_003"}
{"prompt": "Write a function to merge two sorted lists into one sorted list without using built-in sort methods. Use O(n) time complexity.", "test_cases": [{"input": "[1,3,5], [2,4,6]", "expected": "[1,2,3,4,5,6]"}, {"input": "[], [1,2,3]", "expected": "[1,2,3]"}, {"input": "[1,2,3], []", "expected": "[1,2,3]"}], "quality_score": 0.85, "session_id": "train_004"}
{"prompt": "Implement a LRU (Least Recently Used) cache using OrderedDict or dict + linked list. Support get and put operations with O(1) complexity.", "test_cases": [{"input": "LRUCache(2), put(1,1), put(2,2), get(1), put(3,3), get(2)", "expected": "[1, -1]"}, {"input": "LRUCache(1), put(1,1), get(1), put(2,2), get(1), get(2)", "expected": "[1, -1, 2]"}], "quality_score": 0.90, "session_id": "train_005"}
EOF

cat > $DATA_DIR/val.jsonl <<'EOF'
{"prompt": "Write a function to reverse a linked list iteratively. Return the new head.", "test_cases": [{"input": "1->2->3->4->5", "expected": "5->4->3->2->1"}, {"input": "1", "expected": "1"}], "quality_score": 0.87, "session_id": "val_001"}
{"prompt": "Implement binary search to find the first and last position of an element in a sorted array. Return [-1,-1] if not found.", "test_cases": [{"input": "[5,7,7,8,8,10], target=8", "expected": "[3,4]"}, {"input": "[5,7,7,8,8,10], target=6", "expected": "[-1,-1]"}], "quality_score": 0.83, "session_id": "val_002"}
EOF

# 转换为 SkyRL 格式
echo "=== 转换数据格式 ==="
cd /dstack/persistent/rl-experiment

# 克隆我们的 RL 训练仓库获取数据处理器
git clone https://github.com/zeyuyuyu/0g-rl-training.git 2>/dev/null || true
cd 0g-rl-training

# 使用 SkyRL 专用数据处理器
python3 src/skyrl_data_processor.py \
    --input $DATA_DIR/train.jsonl \
    --output-dir $DATA_DIR/processed \
    --env-class opencode \
    --min-quality 0.4 \
    --val-ratio 0.1

echo "数据准备完成"

# =============================================================================
# 4. 创建 SkyRL 配置文件
# =============================================================================
echo "=== 创建 SkyRL 配置 ==="

cat > $WORK_DIR/skyrl_qwen25_7b_config.sh <<'CONFIG'
#!/bin/bash
# SkyRL GRPO Training Config for Qwen2.5-7B

set -x

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# Data configuration (使用处理后的 parquet 文件)
TRAIN_DATA="/dstack/persistent/rl-experiment/data/processed/train.parquet"
VAL_DATA="/dstack/persistent/rl-experiment/data/processed/val.parquet"

# Training configuration
ADV_ESTIMATOR="grpo"
BATCH_SIZE=8
ROLLOUT_N=4
LEARNING_RATE=1e-6
TOTAL_STEPS=100
SAVE_FREQ=50
EVAL_FREQ=25

# GPU configuration
N_GPUS=1
TP_SIZE=1

# Output
OUTPUT_DIR="/dstack/persistent/rl-experiment/outputs"
PROJECT_NAME="0g-skyrl-qwen25-7b"
RUN_NAME="grpo-test-$(date +%Y%m%d-%H%M%S)"

mkdir -p $OUTPUT_DIR

# Launch training with SkyRL
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="[$TRAIN_DATA]" \
    data.val_data="[$VAL_DATA]" \
    trainer.algorithm.advantage_estimator=$ADV_ESTIMATOR \
    trainer.policy.model.path=$MODEL_PATH \
    trainer.policy.model.enable_gradient_checkpointing=true \
    trainer.optimizer.lr=$LEARNING_RATE \
    trainer.train_batch_size=$BATCH_SIZE \
    trainer.total_steps=$TOTAL_STEPS \
    trainer.strategy=fsdp2 \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_gpus_per_node=$N_GPUS \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=$TP_SIZE \
    generator.inference_engine.gpu_memory_utilization=0.6 \
    trainer.rollout.n=$ROLLOUT_N \
    trainer.ckpt_interval=$SAVE_FREQ \
    trainer.eval_interval=$EVAL_FREQ \
    trainer.eval_before_train=true \
    trainer.output_dir=$OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.run_name=$RUN_NAME \
    trainer.logger=console \
    environment.env_class=opencode \
    "$@"
CONFIG

chmod +x $WORK_DIR/skyrl_qwen25_7b_config.sh

# =============================================================================
# 5. 预下载模型
# =============================================================================
echo "=== 预下载模型 ==="
export HF_HOME="$MODEL_CACHE/hf"
export TRANSFORMERS_CACHE="$MODEL_CACHE/transformers"
mkdir -p $HF_HOME $TRANSFORMERS_CACHE

python3 <<'PYEOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Downloading {model_id}...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Tokenizer downloaded")

# Download model weights (CPU to save VRAM for training)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
print("Model weights downloaded")
print("Model size:", sum(p.numel() for p in model.parameters()) / 1e9, "B parameters")
PYEOF

# =============================================================================
# 6. 启动训练
# =============================================================================
echo "=== 启动 SkyRL 训练 ==="
echo "配置文件: $WORK_DIR/skyrl_qwen25_7b_config.sh"
echo "日志将输出到: $OUTPUT_DIR"

# 执行训练
bash $WORK_DIR/skyrl_qwen25_7b_config.sh

echo "=== 训练完成 ==="
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "查看结果:"
echo "  ls -la $OUTPUT_DIR"
