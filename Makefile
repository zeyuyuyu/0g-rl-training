.PHONY: install data train train-test eval export test clean

install:
	pip install -r requirements.txt
	pip install verl
	pip install vllm==0.8.2
	pip install pycryptodome

data:
	python src/data_processor.py \
		--input ./raw/sessions.jsonl \
		--output-dir ./data \
		--format opencode \
		--min-quality 0.4

train:
	./scripts/train_grpo.sh --gpus 4 --model Qwen/Qwen3.5-35B-A3B --epochs 5

train-test:
	./scripts/train_grpo.sh --local-test

eval:
	python src/evaluate.py \
		--model ./checkpoints/grpo/final \
		--test-data ./data/val.jsonl \
		--use-vllm \
		--output ./eval_results

export:
	python scripts/export_for_0g.py \
		--checkpoint ./checkpoints/grpo/final \
		--output ./models/0g_ready \
		--compute-hash \
		--generate-config

test:
	python -m pytest tests/ -v

clean:
	rm -rf ./checkpoints/test ./logs/__pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
