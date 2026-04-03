#!/bin/bash
# GRPO Training Script — veRL Official CLI Interface
#
# veRL uses `python3 -m verl.trainer.main_ppo` with dot-notation Hydra overrides.
# This script wraps the official interface with sensible defaults for our use case.
#
# Usage:
#   ./scripts/train_grpo.sh                   # Default: 4 GPUs, Qwen3.5-35B-A3B
#   ./scripts/train_grpo.sh --gpus 8          # 8 GPUs
#   ./scripts/train_grpo.sh --model Qwen/Qwen2.5-7B-Instruct --gpus 4
#   ./scripts/train_grpo.sh --local-test      # Quick local test with 0.5B model
#
# Reference:
#   veRL examples: https://github.com/verl-project/verl/tree/main/examples/grpo_trainer
#   Qwen3.5-SWE:  https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1

set -x

# ---------------------------------------------------------------------------
# Defaults (override via env vars or CLI flags)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
TRAIN_DATA="${TRAIN_DATA:-./data/train.parquet}"
VAL_DATA="${VAL_DATA:-./data/val.parquet}"
N_GPUS="${N_GPUS:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"
LR="${LR:-5e-6}"
PROJECT="${PROJECT:-0g-rl-training}"
EXPERIMENT="${EXPERIMENT:-qwen35-grpo}"
REWARD_FN="${REWARD_FN:-./src/reward_functions.py}"
REWARD_FN_NAME="${REWARD_FN_NAME:-compute_score}"
TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
LOCAL_TEST=false

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL_PATH="$2";  shift 2 ;;
        --gpus)         N_GPUS="$2";      shift 2 ;;
        --rollout-n)    ROLLOUT_N="$2";   shift 2 ;;
        --epochs)       TOTAL_EPOCHS="$2"; shift 2 ;;
        --batch)        TRAIN_BATCH="$2"; shift 2 ;;
        --lr)           LR="$2";          shift 2 ;;
        --tp)           TP_SIZE="$2";     shift 2 ;;
        --train-data)   TRAIN_DATA="$2";  shift 2 ;;
        --val-data)     VAL_DATA="$2";    shift 2 ;;
        --project)      PROJECT="$2";     shift 2 ;;
        --experiment)   EXPERIMENT="$2";  shift 2 ;;
        --reward-fn)    REWARD_FN="$2";   shift 2 ;;
        --local-test)   LOCAL_TEST=true;  shift 1 ;;
        *)              echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Local test overrides (small model, tiny batch, 1 epoch)
# ---------------------------------------------------------------------------
if [ "$LOCAL_TEST" = true ]; then
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
    TRAIN_BATCH=8
    ROLLOUT_N=2
    TOTAL_EPOCHS=1
    N_GPUS=1
    TP_SIZE=1
    LR="1e-5"
    EXPERIMENT="local-test"
    echo "=== LOCAL TEST MODE ==="
fi

echo "============================================="
echo "  GRPO Training — 0G Compute Network"
echo "============================================="
echo "  Model:       $MODEL_PATH"
echo "  GPUs:        $N_GPUS"
echo "  Rollout N:   $ROLLOUT_N"
echo "  Batch Size:  $TRAIN_BATCH"
echo "  LR:          $LR"
echo "  TP Size:     $TP_SIZE"
echo "  Epochs:      $TOTAL_EPOCHS"
echo "  Reward Fn:   $REWARD_FN"
echo "============================================="

# ---------------------------------------------------------------------------
# Verify environment
# ---------------------------------------------------------------------------
if ! python3 -c "import verl" 2>/dev/null; then
    echo "ERROR: veRL not installed. Run: pip install verl"
    exit 1
fi

if ! python3 -c "import vllm" 2>/dev/null && [ "$ROLLOUT_ENGINE" = "vllm" ]; then
    echo "ERROR: vLLM not installed. Run: pip install vllm"
    exit 1
fi

mkdir -p ./checkpoints ./logs

# ---------------------------------------------------------------------------
# Launch training
#
# This directly follows veRL's official interface:
#   python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
#
# All parameters use veRL's dot-notation config keys.
# See: https://verl.readthedocs.io/en/v0.5.x/algo/grpo.html
# ---------------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=8192 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=$ROLLOUT_ENGINE \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    reward.custom_reward_function.path=$REWARD_FN \
    reward.custom_reward_function.name=$REWARD_FN_NAME \
    \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=$PROJECT \
    trainer.experiment_name=$EXPERIMENT \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@"
