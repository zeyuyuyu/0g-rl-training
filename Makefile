# Makefile for RL Training
# Simplified commands for daily operations

.PHONY: help install data train eval export clean

# Default config
CONFIG ?= configs/grpo_qwen.yaml
EXPERIMENT ?= qwen-grpo-256k

help: ## Show this help message
	@echo "RL Training for 0G Platform"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

data: ## Prepare training data (example)
	python src/data_processor.py \
		--input ./raw/sessions.jsonl \
		--output-dir ./data \
		--format opencode
	@echo "✓ Data prepared in ./data/"

train: ## Start single GPU training
	./scripts/train_grpo.sh single

train-multi: ## Start multi-GPU training (4 GPUs)
	./scripts/train_grpo.sh multi 4

train-0g: ## Train with 0G export enabled
	./scripts/train_grpo.sh 0g

eval: ## Evaluate latest checkpoint
	python src/evaluate.py \
		--model ./checkpoints/grpo/final \
		--test-data ./data/val.jsonl \
		--use-vllm \
		--output ./eval_results

export: ## Export model for 0G deployment
	python scripts/export_for_0g.py \
		--checkpoint ./checkpoints/grpo/final \
		--output ./models/0g_ready \
		--compute-hash \
		--generate-config

quick-test: ## Quick test with small data
	@echo "Running quick test..."
	python -c "import verl; print('veRL OK')"
	python -c "from src.reward_functions import CodeExecutionReward; print('Reward function OK')"
	python -c "from src.data_processor import RLDataPoint; print('Data processor OK')"
	@echo "✓ All imports successful"

clean: ## Clean temporary files
	rm -rf ./logs/*.log
	rm -rf ./checkpoints/temp*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned temporary files"

# Development helpers
format: ## Format code with black
	black src/ scripts/

lint: ## Run type checking
	mypy src/ --ignore-missing-imports

test: ## Run unit tests
	pytest tests/ -v

# Advanced: Full pipeline
full-pipeline: data train export ## Run full pipeline: data → train → export
	@echo "✓ Full pipeline complete!"
