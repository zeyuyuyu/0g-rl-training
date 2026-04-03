"""
Data Processing for RL Training

Converts various data formats to veRL-compatible parquet format.
Supports:
- Coding agent trajectories (Codex, Claude Code, OpenCode sessions)
- Multi-turn conversations
- SFT data with quality scores

Reference: https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import argparse


@dataclass
class RLDataPoint:
    """
    Single data point for RL training.
    
    For GRPO, each data point contains:
    - prompt: The input instruction
    - reference_completions: Optional reference answers (for SFT warmup)
    - test_cases: Optional test cases for code execution reward
    - metadata: Additional info (source, difficulty, etc.)
    """
    prompt: str
    task_id: str
    reference_completions: Optional[List[str]] = None
    test_cases: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame (veRL format)
        
        Note: veRL expects specific column names. For GRPO training:
        - prompt: the input text
        - response: optional reference response
        - data_source: identifier for the dataset
        - reward_model: optional reward model config (not used in our case)
        """
        # Store test_cases in metadata since veRL doesn't have a dedicated column
        meta = self.metadata or {}
        if self.test_cases:
            meta["test_cases"] = self.test_cases
        
        # Primary response (first reference completion)
        response = None
        if self.reference_completions and len(self.reference_completions) > 0:
            response = self.reference_completions[0]
        
        return {
            "prompt": self.prompt,
            "response": response,  # veRL expects this column
            "data_source": meta.get("source", "unknown"),
            "prompt_metadata": json.dumps(meta),  # Store all metadata as JSON string
        }


class CodingDataProcessor:
    """
    Processor for coding agent trajectories.
    
    Input formats supported:
    - OpenCode sessions (JSONL)
    - Claude Code exports (JSON)
    - Custom format with (prompt, code, tests)
    """
    
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_from_opencode_sessions(
        self,
        input_file: str,
        min_quality_score: float = 0.5
    ) -> List[RLDataPoint]:
        """
        Extract training pairs from OpenCode sessions.
        
        Format expected:
        {
            "session_id": "...",
            "prompt": "Write a function...",
            "response": "```python\ndef...",
            "quality_score": 0.8,
            "test_cases": [...],
            "final_code": "..."
        }
        """
        data_points = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing OpenCode sessions"):
                try:
                    session = json.loads(line.strip())
                    
                    # Filter by quality
                    if session.get("quality_score", 0) < min_quality_score:
                        continue
                    
                    # Build prompt
                    prompt = self._build_coding_prompt(
                        instruction=session.get("prompt", ""),
                        context=session.get("context", "")
                    )
                    
                    # Extract reference completion
                    reference = session.get("final_code") or session.get("response", "")
                    
                    # Parse test cases
                    test_cases = session.get("test_cases", [])
                    
                    data_points.append(RLDataPoint(
                        prompt=prompt,
                        task_id=session.get("session_id", "unknown"),
                        reference_completions=[reference] if reference else None,
                        test_cases=test_cases,
                        metadata={
                            "source": "opencode",
                            "quality_score": session.get("quality_score"),
                            "num_turns": session.get("num_turns", 1),
                        }
                    ))
                    
                except Exception as e:
                    print(f"Error processing session: {e}")
                    continue
        
        print(f"Extracted {len(data_points)} data points from {input_file}")
        return data_points
    
    def extract_from_multiturn_sessions(
        self,
        input_file: str,
        max_turns: int = 50
    ) -> List[RLDataPoint]:
        """
        Extract from multi-turn agent conversations.
        
        Format:
        {
            "conversation_id": "...",
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "tool_calls": [...]},
                {"role": "tool", "content": "...", "success": true},
                ...
            ],
            "final_answer": "...",
            "success": true
        }
        """
        data_points = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing multi-turn sessions"):
                try:
                    session = json.loads(line.strip())
                    messages = session.get("messages", [])
                    
                    # Truncate if too long
                    if len(messages) > max_turns:
                        messages = messages[:max_turns]
                    
                    # Build prompt from conversation history
                    prompt = self._conversation_to_prompt(messages)
                    
                    # Use final answer as reference
                    reference = session.get("final_answer", "")
                    
                    data_points.append(RLDataPoint(
                        prompt=prompt,
                        task_id=session.get("conversation_id", "unknown"),
                        reference_completions=[reference] if reference else None,
                        metadata={
                            "source": "multiturn",
                            "num_turns": len(messages),
                            "success": session.get("success", False),
                        }
                    ))
                    
                except Exception as e:
                    print(f"Error processing conversation: {e}")
                    continue
        
        print(f"Extracted {len(data_points)} multi-turn sessions")
        return data_points
    
    def create_from_sft_data(
        self,
        sft_data: List[Dict[str, Any]],
        label_with_claude: bool = False
    ) -> List[RLDataPoint]:
        """
        Convert SFT data to RL format with quality labels.
        
        Reference: Qwen3.5 uses Claude Opus 4.6 for quality scoring
        with avg reward 0.477
        """
        data_points = []
        
        for item in tqdm(sft_data, desc="Processing SFT data"):
            # Extract prompt and completion
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            
            # Quality score (use provided or compute heuristic)
            if "quality_score" in item:
                quality = item["quality_score"]
            elif label_with_claude:
                # Would call Claude API here for labeling
                quality = 0.5  # Placeholder
            else:
                # Heuristic quality based on length/format
                quality = self._heuristic_quality(completion)
            
            # Filter low quality (Qwen3.5 uses avg reward 0.477 as threshold)
            if quality < 0.4:
                continue
            
            data_points.append(RLDataPoint(
                prompt=prompt,
                task_id=item.get("id", "unknown"),
                reference_completions=[completion],
                test_cases=item.get("test_cases"),
                metadata={
                    "source": "sft",
                    "quality_score": quality,
                }
            ))
        
        print(f"Created {len(data_points)} RL data points from SFT")
        return data_points
    
    def _build_coding_prompt(self, instruction: str, context: str = "") -> str:
        """Build a standardized coding prompt"""
        system_msg = (
            "You are an expert coding assistant. "
            "Write clean, efficient, and well-documented Python code."
        )
        
        if context:
            prompt = f"{system_msg}\n\nContext: {context}\n\nTask: {instruction}\n\n"
        else:
            prompt = f"{system_msg}\n\nTask: {instruction}\n\n"
        
        prompt += "Provide your solution in a Python code block:"
        return prompt
    
    def _conversation_to_prompt(self, messages: List[Dict]) -> str:
        """Convert conversation messages to prompt string"""
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
                # Include tool calls info
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    lines.append(f"[Tool calls: {len(tool_calls)}]")
            elif role == "tool":
                lines.append(f"Tool result: {content}")
        
        return "\n".join(lines)
    
    def _heuristic_quality(self, completion: str) -> float:
        """Heuristic quality score based on completion features"""
        score = 0.5  # Base score
        
        # Has code block
        if "```python" in completion or "```" in completion:
            score += 0.1
        
        # Has explanation + code
        if "def " in completion and len(completion) > 200:
            score += 0.1
        
        # Has docstring
        if '"""' in completion or "'''" in completion:
            score += 0.1
        
        # Reasonable length (not too short or too long)
        length = len(completion)
        if 100 < length < 2000:
            score += 0.1
        
        # Has imports (real code)
        if "import " in completion:
            score += 0.1
        
        return min(1.0, score)
    
    def save_to_parquet(
        self,
        data_points: List[RLDataPoint],
        output_file: str,
        split: str = "train"
    ):
        """Save data points to parquet file (veRL format)"""
        output_path = self.output_dir / f"{split}.parquet"
        
        # Convert to DataFrame
        records = [dp.to_dict() for dp in data_points]
        df = pd.DataFrame(records)
        
        # Save as parquet
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        print(f"Saved {len(df)} records to {output_path}")
        
        # Print statistics
        print(f"\nDataset statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  With reference: {df['reference_completions'].notna().sum()}")
        print(f"  With test cases: {df['test_cases'].notna().sum()}")
        
        return output_path
    
    def create_train_val_split(
        self,
        data_points: List[RLDataPoint],
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        """Create train/val split"""
        import random
        random.seed(seed)
        
        # Shuffle
        shuffled = data_points.copy()
        random.shuffle(shuffled)
        
        # Split
        val_size = max(1, int(len(shuffled) * val_ratio))  # At least 1 sample
        val_data = shuffled[:val_size]
        train_data = shuffled[val_size:]
        
        if len(train_data) == 0:
            print(f"Warning: Training set is empty! Using all data for training.")
            train_data = shuffled
            val_data = shuffled[:max(1, min(10, len(shuffled) // 10))]  # Small val set
        
        # Save
        self.save_to_parquet(train_data, "train.parquet", "train")
        self.save_to_parquet(val_data, "val.parquet", "val")
        
        return train_data, val_data


def prepare_qwen35_style_data():
    """
    Create dataset similar to Qwen3.5-35B-A3B-Turbo-SWE
    
    Steps:
    1. Data Extraction: 4,551 training pairs from 4,756 sessions
    2. Labeling: 3,580 pairs with quality scores (avg reward 0.477)
    3. Filter: Keep high-quality pairs for training
    """
    processor = CodingDataProcessor()
    
    # Example: Process multiple sources
    all_data = []
    
    # 1. From OpenCode sessions
    # all_data.extend(processor.extract_from_opencode_sessions(
    #     "./raw/opencode_sessions.jsonl",
    #     min_quality_score=0.4
    # ))
    
    # 2. From multi-turn sessions
    # all_data.extend(processor.extract_from_multiturn_sessions(
    #     "./raw/multiturn_sessions.jsonl"
    # ))
    
    # 3. From SFT data
    # sft_data = json.load(open("./raw/sft_data.json"))
    # all_data.extend(processor.create_from_sft_data(sft_data))
    
    print(f"Total data points: {len(all_data)}")
    
    # Create train/val split
    train_data, val_data = processor.create_train_val_split(all_data, val_ratio=0.1)
    
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Process data for RL training")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--format", type=str, choices=["opencode", "multiturn", "sft"],
                        default="opencode", help="Input format")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--min-quality", type=float, default=0.4, help="Minimum quality score")
    
    args = parser.parse_args()
    
    processor = CodingDataProcessor(output_dir=args.output_dir)
    
    # Process based on format
    if args.format == "opencode":
        data = processor.extract_from_opencode_sessions(
            args.input,
            min_quality_score=args.min_quality
        )
    elif args.format == "multiturn":
        data = processor.extract_from_multiturn_sessions(args.input)
    elif args.format == "sft":
        with open(args.input, 'r') as f:
            sft_data = [json.loads(line) for line in f]
        data = processor.create_from_sft_data(sft_data)
    
    # Create split
    processor.create_train_val_split(data, val_ratio=args.val_ratio)
    
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
