"""
Unit tests for veRL-compatible reward functions.
Tests the compute_score interface that veRL calls.
"""

import json
import pytest

from src.reward_functions import compute_score, _extract_code, _check_syntax, _check_style


class TestComputeScoreInterface:
    """Test the top-level compute_score function (veRL entry point)."""

    def test_coding_good_code(self):
        solution = '```python\ndef add(a, b):\n    """Add two numbers."""\n    return a + b\n```'
        result = compute_score(
            data_source="coding",
            solution_str=solution,
            ground_truth="",
            extra_info={},
        )
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["compile"] == 1.0

    def test_coding_syntax_error(self):
        solution = '```python\ndef add(a, b\n    return a + b\n```'
        result = compute_score(
            data_source="coding",
            solution_str=solution,
            ground_truth="",
            extra_info={},
        )
        assert result["compile"] == 0.0
        assert result["score"] < 0.5

    def test_coding_no_code(self):
        result = compute_score(
            data_source="coding",
            solution_str="I think you should just think about it.",
            ground_truth="",
        )
        assert result["score"] == 0.0
        assert result["reason"] == "no_code_found"

    def test_coding_with_test_cases(self):
        solution = '```python\ndef add(a, b):\n    return a + b\nprint(add(1, 2))\n```'
        test_cases = [{"input": None, "expected": "3\n"}]
        result = compute_score(
            data_source="coding",
            solution_str=solution,
            ground_truth=json.dumps(test_cases),
            extra_info={},
        )
        assert result["score"] > 0.0
        assert result["compile"] == 1.0

    def test_agent_trajectory(self):
        trajectory = json.dumps([
            {"role": "assistant", "content": "Let me search", "tool_calls": [{"name": "grep"}]},
            {"role": "tool", "content": "found", "success": True},
            {"role": "assistant", "content": "Task completed successfully!"},
        ])
        result = compute_score(
            data_source="agent",
            solution_str=trajectory,
            ground_truth="",
            extra_info={},
        )
        assert "score" in result
        assert result["score"] > 0.5
        assert result["tool_success"] == 1.0
        assert result["task_completion"] == 1.0

    def test_generic_fallback(self):
        result = compute_score(
            data_source="unknown_dataset",
            solution_str="Some random text",
            ground_truth="",
        )
        assert "score" in result
        assert result["method"] == "heuristic"

    def test_generic_exact_match(self):
        result = compute_score(
            data_source="math",
            solution_str="The answer is 42.",
            ground_truth="42",
        )
        assert result["score"] == 1.0
        assert result["method"] == "exact_match"


class TestCodeExtraction:
    def test_python_block(self):
        text = "Here:\n```python\nprint('hi')\n```\nDone."
        assert _extract_code(text) == "print('hi')"

    def test_plain_block(self):
        text = "Here:\n```\nimport os\nprint(os.getcwd())\n```"
        code = _extract_code(text)
        assert code is not None
        assert "import os" in code

    def test_raw_code(self):
        text = "def hello():\n    return 'world'"
        assert _extract_code(text) is not None

    def test_no_code(self):
        assert _extract_code("Just a plain sentence.") is None


class TestSyntaxCheck:
    def test_valid(self):
        ok, err = _check_syntax("x = 1 + 2")
        assert ok and err == ""

    def test_invalid(self):
        ok, err = _check_syntax("def f(\n  return 1")
        assert not ok


class TestStyleCheck:
    def test_good_style(self):
        code = 'def add(a, b):\n    """Add."""\n    return a + b'
        assert _check_style(code) >= 0.8

    def test_missing_docstring(self):
        code = "def add(a, b):\n    return a + b"
        score = _check_style(code)
        assert score < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
