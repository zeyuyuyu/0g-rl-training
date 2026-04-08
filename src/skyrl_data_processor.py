"""
SkyRL Data Processor

将 coding sessions 转换为 SkyRL 训练格式。

SkyRL 数据格式（Parquet）:
- prompt: str - 完整的 prompt 文本
- ability: str - 任务类型 (coding, agent, etc.)
- env_class: str - 环境类名 (opencode, gsm8k, etc.)
- reward_model: dict - {"style": "rule", "ground_truth": ...}
- extra_info: dict - 额外元数据

与 veRL 的主要区别：
- SkyRL prompt 是纯文本字符串（不是 chat template list）
- 需要 env_class 字段来指定环境类型
"""

import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "You are an expert coding assistant. "
    "Write clean, efficient, and well-documented code. "
    "Think step by step and provide your solution in a Python code block."
)


def convert_to_skyrl_format(
    example: Dict[str, Any],
    idx: int,
    split: str,
    env_class: str = "opencode"
) -> Dict[str, Any]:
    """
    将原始数据转换为 SkyRL 格式。
    
    SkyRL 与 veRL 的主要区别：
    1. prompt 是纯文本字符串（不是 list[dict]）
    2. 需要 env_class 字段
    3. reward_model.ground_truth 存储 test_cases 用于沙箱执行
    """
    instruction = example.get("prompt", "")
    context = example.get("context", "")
    test_cases = example.get("test_cases", [])
    
    # 构建纯文本 prompt（不是 chat template）
    if context:
        prompt_text = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nTask: {instruction}"
    else:
        prompt_text = f"{SYSTEM_PROMPT}\n\nTask: {instruction}"
    
    # 构建 reward_model
    reward_model = {
        "style": "rule",
        "ground_truth": json.dumps({
            "test_cases": test_cases,
            "verification_type": "execution"
        }) if test_cases else ""
    }
    
    # 构建 extra_info
    extra_info = {
        "split": split,
        "index": idx,
        "session_id": example.get("session_id", f"{split}_{idx}"),
        "quality_score": example.get("quality_score", 0.5),
        "num_test_cases": len(test_cases)
    }
    
    return {
        "prompt": prompt_text,
        "ability": "coding",
        "env_class": env_class,
        "reward_model": reward_model,
        "extra_info": extra_info
    }


def process_for_skyrl(
    input_path: str,
    output_dir: str,
    env_class: str = "opencode",
    min_quality: float = 0.0,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    处理数据并保存为 SkyRL 格式的 Parquet。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    
    # 读取 JSONL
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    ds = datasets.Dataset.from_list(records)
    print(f"  Loaded {len(ds)} records")
    
    # 质量过滤
    if min_quality > 0 and "quality_score" in ds.column_names:
        ds = ds.filter(lambda x: x.get("quality_score", 0) >= min_quality)
        print(f"  After quality filter (>= {min_quality}): {len(ds)} records")
    
    # 划分 train/val
    ds = ds.shuffle(seed=seed)
    val_size = max(1, int(len(ds) * val_ratio))
    train_ds = ds.select(range(val_size, len(ds)))
    val_ds = ds.select(range(val_size))
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # 转换为 SkyRL 格式
    def make_map_fn(split: str):
        def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            return convert_to_skyrl_format(example, idx, split, env_class)
        return process_fn
    
    train_ds = train_ds.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=train_ds.column_names
    )
    
    val_ds = val_ds.map(
        function=make_map_fn("val"),
        with_indices=True,
        remove_columns=val_ds.column_names
    )
    
    # 保存为 Parquet
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    
    print(f"  Saved: {train_path} ({len(train_ds)} rows)")
    print(f"  Saved: {val_path} ({len(val_ds)} rows)")
    
    # 验证
    _verify_skyrl_parquet(train_path)
    
    return train_path, val_path


def _verify_skyrl_parquet(path: str):
    """验证 Parquet 包含 SkyRL 所需的字段。"""
    table = pq.read_table(path)
    required = {"prompt", "ability", "env_class", "reward_model", "extra_info"}
    actual = set(table.column_names)
    missing = required - actual
    
    if missing:
        print(f"  WARNING: Missing required fields: {missing}")
    else:
        print(f"  Parquet schema OK: {sorted(actual)}")
    
    # 显示 sample
    sample = {col: table.column(col)[0].as_py() for col in actual}
    print(f"  Sample prompt length: {len(sample.get('prompt', ''))} chars")
    print(f"  Sample env_class: {sample.get('env_class', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert data to SkyRL format"
    )
    parser.add_argument("--input", type=str, required=True, 
                        help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, default="./data/skyrl",
                        help="Output directory")
    parser.add_argument("--env-class", type=str, default="opencode",
                        choices=["opencode", "coding", "swe", "agent"],
                        help="Environment class for SkyRL")
    parser.add_argument("--min-quality", type=float, default=0.0,
                        help="Minimum quality score for filtering")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    process_for_skyrl(
        input_path=args.input,
        output_dir=args.output_dir,
        env_class=args.env_class,
        min_quality=args.min_quality,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    print("\nDone. Use the parquet files with SkyRL training.")


if __name__ == "__main__":
    main()
