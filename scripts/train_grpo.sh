#!/bin/bash
# GRPO Training Script for Qwen Models
# 
# Usage:
#   Single GPU:     ./train_grpo.sh single
#   Multi-GPU:      ./train_grpo.sh multi 4
#   SLURM cluster:  ./train_grpo.sh slurm
#
# Reference: https://verl.readthedocs.io/en/v0.5.x/algo/grpo.html

set -e

# Configuration
CONFIG_FILE=${CONFIG_FILE:-"./configs/grpo_qwen.yaml"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen-grpo-256k"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints"}
LOG_DIR=${LOG_DIR:-"./logs"}

# Parse command
COMMAND=${1:-"single"}
NUM_GPUS=${2:-4}

echo "=========================================="
echo "GRPO Training for 0G Platform"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Command: $COMMAND"
echo "=========================================="

# Create directories
mkdir -p $OUTPUT_DIR/$EXPERIMENT_NAME
mkdir -p $LOG_DIR

# Check if veRL is installed
if ! python -c "import verl" 2>/dev/null; then
    echo "Error: veRL not found. Please install: pip install verl"
    exit 1
fi

# Training function
train_single() {
    echo "Running single GPU training..."
    
    python -m verl.trainer.main_ppo \
        config_path=$CONFIG_FILE \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.checkpoint.output_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
        trainer.logging.log_dir=$LOG_DIR \
        algorithm.adv_estimator=grpo \
        2>&1 | tee $LOG_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log
}

train_multi() {
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    
    # Use torchrun for multi-GPU
    torchrun --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        -m verl.trainer.main_ppo \
        config_path=$CONFIG_FILE \
        trainer.experiment_name=${EXPERIMENT_NAME}_multi \
        trainer.checkpoint.output_dir=$OUTPUT_DIR/${EXPERIMENT_NAME}_multi \
        trainer.logging.log_dir=$LOG_DIR \
        algorithm.adv_estimator=grpo \
        2>&1 | tee $LOG_DIR/${EXPERIMENT_NAME}_multi_$(date +%Y%m%d_%H%M%S).log
}

train_slurm() {
    echo "Submitting SLURM job..."
    
    # SLURM script
    cat > /tmp/grpo_slurm_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=grpo-training
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

# Load modules
module load cuda/12.1
module load python/3.10

# Activate environment
source ~/verl_env/bin/activate

# Get node info
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node ($head_node_ip)"
echo "Nodes: ${nodes_array[@]}"

# Launch training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --master_addr=$head_node_ip \
    --master_port=29500 \
    -m verl.trainer.main_ppo \
    config_path=./configs/grpo_qwen.yaml \
    algorithm.adv_estimator=grpo \
    trainer.experiment_name=grpo_slurm_$SLURM_JOB_ID
EOF

    sbatch /tmp/grpo_slurm_job.sh
    echo "SLURM job submitted. Check: squeue -u $USER"
}

train_0g_deploy() {
    echo "Training with 0G deployment config..."
    
    # Special config for 0G platform
    # - Freeze router for MoE
    # - LoRA export enabled
    # - Model hash calculation
    
    python -m verl.trainer.main_ppo \
        config_path=$CONFIG_FILE \
        trainer.experiment_name=${EXPERIMENT_NAME}_0g \
        actor_rollout.actor.freeze_modules=[router,gate] \
        platform.export_for_0g=true \
        2>&1 | tee $LOG_DIR/${EXPERIMENT_NAME}_0g_$(date +%Y%m%d_%H%M%S).log
    
    # After training, export for 0G
    echo "Exporting model for 0G deployment..."
    python scripts/export_for_0g.py \
        --checkpoint $OUTPUT_DIR/${EXPERIMENT_NAME}_0g \
        --output ./models/0g_ready_model \
        --compute-hash
}

# Main execution
case $COMMAND in
    single)
        train_single
        ;;
    multi)
        train_multi
        ;;
    slurm)
        train_slurm
        ;;
    0g)
        train_0g_deploy
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: $0 {single|multi|slurm|0g} [num_gpus]"
        exit 1
        ;;
esac

echo "=========================================="
echo "Training complete!"
echo "Checkpoints: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "Logs: $LOG_DIR"
echo "=========================================="
