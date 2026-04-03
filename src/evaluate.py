"""
Evaluation Suite for RL-Trained Models

Provides comprehensive evaluation metrics:
1. Code execution accuracy (compile + test pass rate)
2. Reward distribution analysis
3. Comparison with SFT baseline
4. Long-context retention tests

Reference: https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from reward_functions import CodeExecutionReward, RewardResult


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    total_samples: int
    avg_reward: float
    compile_success_rate: float
    test_pass_rate: float
    style_score_avg: float
    reward_distribution: Dict[str, float]
    per_task_scores: List[Dict[str, Any]]


class RLModelEvaluator:
    """
    Evaluator for RL-trained models.
    
    Supports both:
    - vLLM for fast batch inference
    - HF Transformers for single-sample debugging
    """
    
    def __init__(
        self,
        model_path: str,
        use_vllm: bool = True,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768
    ):
        self.model_path = model_path
        self.use_vllm = use_vllm
        
        print(f"Loading model from {model_path}...")
        
        if use_vllm:
            # vLLM for fast inference
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                dtype="bfloat16"
            )
            self.tokenizer = self.llm.get_tokenizer()
        else:
            # HF Transformers
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = None
        
        # Reward function
        self.reward_fn = CodeExecutionReward()
        
        print("Model loaded successfully")
    
    def evaluate_code_dataset(
        self,
        test_data: List[Dict[str, Any]],
        num_samples: Optional[int] = None,
        temperature: float = 0.6,
        top_p: float = 0.9
    ) -> EvaluationResult:
        """
        Evaluate model on coding tasks.
        
        Args:
            test_data: List of test cases with prompt and test_cases
            num_samples: Number of samples to evaluate (None = all)
            temperature: Generation temperature
            top_p: Nucleus sampling parameter
        """
        if num_samples:
            test_data = test_data[:num_samples]
        
        print(f"Evaluating on {len(test_data)} samples...")
        
        # Generate completions
        prompts = [item["prompt"] for item in test_data]
        
        if self.use_vllm:
            completions = self._generate_vllm(
                prompts,
                temperature=temperature,
                top_p=top_p
            )
        else:
            completions = self._generate_hf(
                prompts,
                temperature=temperature,
                top_p=top_p
            )
        
        # Compute rewards
        results = []
        for item, completion in tqdm(
            zip(test_data, completions),
            total=len(test_data),
            desc="Computing rewards"
        ):
            # Update reward function with test cases
            if item.get("test_cases"):
                self.reward_fn.test_cases = item["test_cases"]
            
            reward_result = self.reward_fn.compute_reward(completion)
            
            results.append({
                "task_id": item.get("task_id", "unknown"),
                "prompt": item["prompt"],
                "completion": completion,
                "reward": reward_result.total_score,
                "breakdown": reward_result.breakdown,
                "details": reward_result.details
            })
        
        # Aggregate metrics
        return self._aggregate_results(results)
    
    def _generate_vllm(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 2048
    ) -> List[str]:
        """Generate using vLLM"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        completions = []
        for output in outputs:
            text = output.outputs[0].text
            completions.append(text)
        
        return completions
    
    def _generate_hf(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 2048
    ) -> List[str]:
        """Generate using HF Transformers"""
        completions = []
        
        for prompt in tqdm(prompts, desc="Generating"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt
            completion = text[len(prompt):]
            completions.append(completion)
        
        return completions
    
    def _aggregate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Aggregate evaluation metrics"""
        total = len(results)
        
        # Average reward
        rewards = [r["reward"] for r in results]
        avg_reward = np.mean(rewards)
        
        # Compile success rate
        compile_success = sum(
            1 for r in results
            if r["details"].get("syntax_ok", False)
        )
        compile_rate = compile_success / total
        
        # Test pass rate (only for those that compiled)
        test_results = [
            r for r in results
            if r["details"].get("test_total", 0) > 0
        ]
        if test_results:
            total_tests = sum(r["details"]["test_total"] for r in test_results)
            passed_tests = sum(r["details"]["test_passed"] for r in test_results)
            test_rate = passed_tests / total_tests if total_tests > 0 else 0
        else:
            test_rate = 0
        
        # Style score
        style_scores = [r["breakdown"]["style"] for r in results]
        avg_style = np.mean(style_scores)
        
        # Reward distribution
        reward_dist = {
            "0.0-0.2": sum(1 for r in rewards if 0.0 <= r < 0.2) / total,
            "0.2-0.4": sum(1 for r in rewards if 0.2 <= r < 0.4) / total,
            "0.4-0.6": sum(1 for r in rewards if 0.4 <= r < 0.6) / total,
            "0.6-0.8": sum(1 for r in rewards if 0.6 <= r < 0.8) / total,
            "0.8-1.0": sum(1 for r in rewards if 0.8 <= r <= 1.0) / total,
        }
        
        return EvaluationResult(
            total_samples=total,
            avg_reward=avg_reward,
            compile_success_rate=compile_rate,
            test_pass_rate=test_rate,
            style_score_avg=avg_style,
            reward_distribution=reward_dist,
            per_task_scores=results
        )
    
    def test_long_context_retention(
        self,
        context_lengths: List[int] = [4096, 16384, 32768, 65536],
        num_queries: int = 10
    ) -> Dict[str, Any]:
        """
        Test model's ability to retain information in long contexts.
        
        Creates documents with specific facts and queries about them
        to measure accuracy vs context length.
        """
        print("Testing long-context retention...")
        
        results = {}
        
        for max_len in context_lengths:
            print(f"  Testing {max_len} tokens...")
            
            # Create synthetic long document with embedded facts
            document = self._create_test_document(max_len)
            
            # Create queries about specific positions in document
            queries = self._create_retrieval_queries(document, num_queries)
            
            # Test accuracy
            correct = 0
            for query in queries:
                # Generate response
                prompt = f"Document:\n{document}\n\nQuestion: {query['question']}"
                
                if self.use_vllm:
                    response = self._generate_vllm([prompt], max_tokens=512)[0]
                else:
                    response = self._generate_hf([prompt], max_new_tokens=512)[0]
                
                # Check if answer is in response
                if query["answer"].lower() in response.lower():
                    correct += 1
            
            accuracy = correct / len(queries)
            results[max_len] = {
                "accuracy": accuracy,
                "num_queries": num_queries,
                "correct": correct
            }
        
        return results
    
    def _create_test_document(self, target_tokens: int) -> str:
        """Create a synthetic document of approximately target_tokens length"""
        # Simple repetitive structure with numbered facts
        paragraphs = []
        fact_counter = 1
        
        approx_chars = target_tokens * 4  # Rough estimate
        
        while len("\n\n".join(paragraphs)) < approx_chars:
            paragraphs.append(
                f"Section {fact_counter}: The unique identifier for this "
                f"section is TOKEN_{fact_counter:04d}. This is important "
                f"information that should be remembered for later retrieval. "
                f"Additional context: Lorem ipsum dolor sit amet. " * 5
            )
            fact_counter += 1
        
        return "\n\n".join(paragraphs)
    
    def _create_retrieval_queries(
        self,
        document: str,
        num_queries: int
    ) -> List[Dict[str, str]]:
        """Create queries about specific facts in the document"""
        import random
        random.seed(42)
        
        queries = []
        # Extract section numbers from document
        for i in random.sample(range(1, 100), num_queries):
            queries.append({
                "question": f"What is the unique identifier for section {i}?",
                "answer": f"TOKEN_{i:04d}"
            })
        
        return queries
    
    def plot_reward_distribution(
        self,
        result: EvaluationResult,
        output_path: str = "./reward_distribution.png"
    ):
        """Plot reward distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Reward distribution bar chart
        ax = axes[0, 0]
        labels = list(result.reward_distribution.keys())
        values = list(result.reward_distribution.values())
        ax.bar(labels, values)
        ax.set_xlabel("Reward Range")
        ax.set_ylabel("Proportion")
        ax.set_title("Reward Distribution")
        
        # 2. Metric comparison
        ax = axes[0, 1]
        metrics = ["Compile", "Test", "Style"]
        values = [
            result.compile_success_rate,
            result.test_pass_rate,
            result.style_score_avg
        ]
        ax.bar(metrics, values)
        ax.set_ylim([0, 1])
        ax.set_ylabel("Score")
        ax.set_title("Quality Metrics")
        
        # 3. Per-sample reward scatter
        ax = axes[1, 0]
        rewards = [r["reward"] for r in result.per_task_scores]
        ax.scatter(range(len(rewards)), rewards, alpha=0.5)
        ax.axhline(y=result.avg_reward, color='r', linestyle='--', label=f'Mean: {result.avg_reward:.3f}')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Reward")
        ax.set_title("Per-Sample Rewards")
        ax.legend()
        
        # 4. Reward breakdown stacked bar (sample)
        ax = axes[1, 1]
        sample_size = min(50, len(result.per_task_scores))
        samples = result.per_task_scores[:sample_size]
        compile_scores = [r["breakdown"]["compile"] for r in samples]
        test_scores = [r["breakdown"]["test"] for r in samples]
        style_scores = [r["breakdown"]["style"] for r in samples]
        
        x = range(sample_size)
        ax.bar(x, compile_scores, label="Compile", alpha=0.7)
        ax.bar(x, test_scores, bottom=compile_scores, label="Test", alpha=0.7)
        ax.bar(x, style_scores, bottom=[c+t for c,t in zip(compile_scores, test_scores)], label="Style", alpha=0.7)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Score")
        ax.set_title("Reward Breakdown (Sample)")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")


def compare_with_baseline(
    rl_model_path: str,
    sft_model_path: str,
    test_data: List[Dict[str, Any]],
    output_report: str = "./comparison_report.json"
):
    """
    Compare RL-trained model with SFT baseline.
    
    Expected improvements:
    - Code compile rate: +10-25%
    - Test pass rate: +15-35%
    - Average reward: +30-50%
    """
    print("="*60)
    print("Comparing RL model vs SFT baseline")
    print("="*60)
    
    # Evaluate both models
    print("\nEvaluating RL model...")
    rl_evaluator = RLModelEvaluator(rl_model_path)
    rl_results = rl_evaluator.evaluate_code_dataset(test_data)
    
    print("\nEvaluating SFT baseline...")
    sft_evaluator = RLModelEvaluator(sft_model_path)
    sft_results = sft_evaluator.evaluate_code_dataset(test_data)
    
    # Compare
    comparison = {
        "rl_model": {
            "avg_reward": rl_results.avg_reward,
            "compile_rate": rl_results.compile_success_rate,
            "test_pass_rate": rl_results.test_pass_rate,
        },
        "sft_model": {
            "avg_reward": sft_results.avg_reward,
            "compile_rate": sft_results.compile_success_rate,
            "test_pass_rate": sft_results.test_pass_rate,
        },
        "improvements": {
            "reward": rl_results.avg_reward - sft_results.avg_reward,
            "reward_pct": (rl_results.avg_reward - sft_results.avg_reward) / sft_results.avg_reward * 100,
            "compile_rate": rl_results.compile_success_rate - sft_results.compile_success_rate,
            "test_rate": rl_results.test_pass_rate - sft_results.test_pass_rate,
        }
    }
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"RL Model - Avg Reward: {comparison['rl_model']['avg_reward']:.3f}")
    print(f"SFT Model - Avg Reward: {comparison['sft_model']['avg_reward']:.3f}")
    print(f"Improvement: {comparison['improvements']['reward_pct']:.1f}%")
    print()
    print(f"Compile Rate: {comparison['rl_model']['compile_rate']:.1%} vs {comparison['sft_model']['compile_rate']:.1%}")
    print(f"Test Pass Rate: {comparison['rl_model']['test_pass_rate']:.1%} vs {comparison['sft_model']['test_pass_rate']:.1%}")
    
    # Save report
    with open(output_report, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nReport saved to {output_report}")
    
    return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RL-trained model")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--test-data", type=str, required=True, help="Test data JSONL")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to eval")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for inference")
    parser.add_argument("--compare-with", type=str, default=None, help="SFT model for comparison")
    
    args = parser.parse_args()
    
    # Load test data
    test_data = []
    with open(args.test_data, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Create output dir
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Evaluation
    if args.compare_with:
        # Comparison mode
        compare_with_baseline(
            args.model,
            args.compare_with,
            test_data,
            output_report=f"{args.output}/comparison.json"
        )
    else:
        # Single model evaluation
        evaluator = RLModelEvaluator(args.model, use_vllm=args.use_vllm)
        
        # Code evaluation
        results = evaluator.evaluate_code_dataset(
            test_data,
            num_samples=args.num_samples
        )
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total samples: {results.total_samples}")
        print(f"Average reward: {results.avg_reward:.3f}")
        print(f"Compile success rate: {results.compile_success_rate:.1%}")
        print(f"Test pass rate: {results.test_pass_rate:.1%}")
        print(f"Style score avg: {results.style_score_avg:.3f}")
        print("\nReward distribution:")
        for bucket, pct in results.reward_distribution.items():
            print(f"  {bucket}: {pct:.1%}")
        
        # Save results
        with open(f"{args.output}/results.json", 'w') as f:
            json.dump({
                "avg_reward": results.avg_reward,
                "compile_success_rate": results.compile_success_rate,
                "test_pass_rate": results.test_pass_rate,
                "style_score_avg": results.style_score_avg,
                "reward_distribution": results.reward_distribution
            }, f, indent=2)
        
        # Plot
        evaluator.plot_reward_distribution(results, f"{args.output}/reward_dist.png")
        
        # Long context test (optional)
        print("\nTesting long context retention...")
        context_results = evaluator.test_long_context_retention()
        print("\nLong-context retention:")
        for length, metrics in context_results.items():
            print(f"  {length} tokens: {metrics['accuracy']:.1%} accuracy")
        
        with open(f"{args.output}/long_context_results.json", 'w') as f:
            json.dump(context_results, f, indent=2)


if __name__ == "__main__":
    main()
