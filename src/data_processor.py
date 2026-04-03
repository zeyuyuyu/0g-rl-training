"""
Data Processing for veRL GRPO Training

将 coding agent sessions 转化为 veRL 要求的 Parquet 格式。

veRL RLHFDataset 要求每条记录包含以下 5 个字段:
    data_source : str          — 数据来源标识，用于在 RewardManager 中选择 reward function
    prompt      : list[dict]   — HuggingFace chat template 格式 [{"role": "user", "content": "..."}]
    ability     : str          — 任务类别 (如 "coding", "agent")
    reward_model: dict         — {"style": "rule"|"model", "ground_truth": str}
    extra_info  : dict         — 附加信息 (split, index, test_cases 等)

参考:
    veRL 数据文档:  https://verl.readthedocs.io/en/v0.5.x/preparation/prepare_data.html
    veRL GSM8k 示例: examples/data_preprocess/gsm8k.py
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import pyarrow.parquet as pq


# =============================================================================
# Core: make_map_fn — 将原始数据转为 veRL 5-field schema
# =============================================================================

SYSTEM_PROMPT = (
    "You are an expert coding assistant. "
    "Write clean, efficient, and well-documented code. "
    "Think step by step and provide your solution in a Python code block."
)


def make_map_fn_opencode(split: str, data_source: str = "coding"):
    """OpenCode / Codex / Claude Code session → veRL record."""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        instruction = example.get("prompt", "")
        context = example.get("context", "")

        if context:
            content = f"Context:\n{context}\n\nTask: {instruction}"
        else:
            content = instruction

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        test_cases = example.get("test_cases", [])
        ground_truth = json.dumps(test_cases) if test_cases else ""

        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "session_id": example.get("session_id", ""),
                "quality_score": example.get("quality_score", 0.0),
                "test_cases": json.dumps(test_cases) if test_cases else "",
            },
        }
    return process_fn


def make_map_fn_multiturn(split: str, data_source: str = "agent"):
    """Multi-turn agent conversation → veRL record."""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        messages = example.get("messages", [])

        prompt = []
        for msg in messages:
            role = msg.get("role", "user")
            if role in ("user", "system", "assistant"):
                prompt.append({"role": role, "content": msg.get("content", "")})
            elif role == "tool":
                prompt.append({"role": "user",
                               "content": f"[Tool Output]: {msg.get('content', '')}"})

        if not prompt:
            prompt = [{"role": "user", "content": "No prompt available"}]

        ground_truth = json.dumps({
            "success": example.get("success", False),
            "final_answer": example.get("final_answer", ""),
        })

        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "agent",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "conversation_id": example.get("conversation_id", ""),
                "num_turns": len(messages),
            },
        }
    return process_fn


def make_map_fn_sft(split: str, data_source: str = "sft"):
    """SFT instruction-completion pair → veRL record."""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        instruction = example.get("prompt", example.get("instruction", ""))
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        test_cases = example.get("test_cases", [])
        ground_truth = json.dumps(test_cases) if test_cases else ""

        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "quality_score": example.get("quality_score", 0.5),
                "test_cases": json.dumps(test_cases) if test_cases else "",
            },
        }
    return process_fn


# =============================================================================
# Pipeline: load → filter → transform → split → parquet
# =============================================================================

FORMAT_MAP = {
    "opencode": make_map_fn_opencode,
    "multiturn": make_map_fn_multiturn,
    "sft": make_map_fn_sft,
}


def load_jsonl(path: str) -> datasets.Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return datasets.Dataset.from_list(records)


def process_dataset(
    input_path: str,
    output_dir: str,
    fmt: str = "opencode",
    min_quality: float = 0.0,
    val_ratio: float = 0.1,
    seed: int = 42,
    data_source: Optional[str] = None,
):
    """
    End-to-end: JSONL → filtered → veRL parquet.

    Steps (following veRL data_preprocess convention):
    1. Load raw JSONL
    2. Filter by quality_score (if field exists)
    3. Apply make_map_fn to transform into 5-field schema
    4. Split train/val
    5. Save as parquet
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {input_path} (format={fmt}) ...")
    ds = load_jsonl(input_path)
    print(f"  Loaded {len(ds)} records")

    # Filter by quality
    if min_quality > 0 and "quality_score" in ds.column_names:
        ds = ds.filter(lambda x: x.get("quality_score", 0) >= min_quality)
        print(f"  After quality filter (>={min_quality}): {len(ds)} records")

    # Train / val split
    ds = ds.shuffle(seed=seed)
    val_size = max(1, int(len(ds) * val_ratio))
    train_ds = ds.select(range(val_size, len(ds)))
    val_ds = ds.select(range(val_size))
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Transform
    make_fn = FORMAT_MAP[fmt]
    ds_name = data_source or fmt

    train_ds = train_ds.map(
        function=make_fn("train", ds_name),
        with_indices=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        function=make_fn("val", ds_name),
        with_indices=True,
        remove_columns=val_ds.column_names,
    )

    # Save parquet
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    print(f"  Saved: {train_path} ({len(train_ds)} rows)")
    print(f"  Saved: {val_path} ({len(val_ds)} rows)")

    # Quick sanity check
    _verify_parquet(train_path)


def _verify_parquet(path: str):
    """Verify parquet has the 5 required veRL fields."""
    table = pq.read_table(path)
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    actual = set(table.column_names)
    missing = required - actual
    if missing:
        print(f"  WARNING: missing required fields: {missing}")
    else:
        print(f"  Parquet schema OK: {sorted(actual)}")
    print(f"  Sample row 0 data_source: {table.column('data_source')[0].as_py()}")


# =============================================================================
# Qwen3.5-SWE style data preparation recipe
# =============================================================================

def prepare_qwen35_swe_style(
    sessions_dir: str,
    output_dir: str = "./data",
    min_quality: float = 0.4,
):
    """
    Replicate the data pipeline described in Qwen3.5-35B-A3B-Turbo-SWE:
      1. 4,551 training pairs from 4,756 coding agent sessions
      2. 3,580 pairs labeled with Claude Opus 4.6 (avg reward 0.477)
      3. Quality filter: score >= 0.4

    Expects sessions_dir to contain JSONL files with fields:
      session_id, prompt, context, response, quality_score, test_cases, final_code
    """
    import glob
    all_records = []
    for f in glob.glob(os.path.join(sessions_dir, "*.jsonl")):
        with open(f, "r") as fh:
            for line in fh:
                all_records.append(json.loads(line.strip()))

    print(f"Total raw sessions: {len(all_records)}")

    # Write to single JSONL
    tmp_path = os.path.join(output_dir, "_raw_combined.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(tmp_path, "w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")

    process_dataset(
        input_path=tmp_path,
        output_dir=output_dir,
        fmt="opencode",
        min_quality=min_quality,
        data_source="coding",
    )
    os.remove(tmp_path)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert raw data to veRL parquet format (5-field schema)"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument(
        "--format", type=str, choices=list(FORMAT_MAP.keys()),
        default="opencode", help="Input data format",
    )
    parser.add_argument("--data-source", type=str, default=None,
                        help="Override data_source field value")
    parser.add_argument("--min-quality", type=float, default=0.0,
                        help="Minimum quality_score for filtering")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    process_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        fmt=args.format,
        min_quality=args.min_quality,
        val_ratio=args.val_ratio,
        seed=args.seed,
        data_source=args.data_source,
    )
    print("Done.")


if __name__ == "__main__":
    main()
