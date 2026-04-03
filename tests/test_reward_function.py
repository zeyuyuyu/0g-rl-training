"""
Unit tests for reward functions
"""

import pytest
from src.reward_functions import (
    CodeExecutionReward,
    AgentTrajectoryReward,
    RewardResult
)


class TestCodeExecutionReward:
    """Test code execution reward function"""
    
    def test_extract_code_from_markdown(self):
        """Test code extraction from markdown blocks"""
        reward_fn = CodeExecutionReward()
        
        # Test ```python block
        completion = """Here's the solution:

```python
def hello():
    return "world"
```

Hope this helps!"""
        
        code = reward_fn.extract_code(completion)
        assert code is not None
        assert "def hello()" in code
    
    def test_extract_code_from_plain(self):
        """Test extraction from plain code"""
        reward_fn = CodeExecutionReward()
        
        completion = """def hello():
    return "world"
"""
        
        code = reward_fn.extract_code(completion)
        assert code is not None
        assert "def hello()" in code
    
    def test_no_code_found(self):
        """Test handling of completion without code"""
        reward_fn = CodeExecutionReward()
        
        completion = "I don't have any code for you."
        
        code = reward_fn.extract_code(completion)
        assert code is None
    
    def test_syntax_check_valid(self):
        """Test syntax checking for valid code"""
        reward_fn = CodeExecutionReward()
        
        code = """def hello():
    return "world"
"""
        
        is_valid, error = reward_fn.check_syntax(code)
        assert is_valid
        assert error == ""
    
    def test_syntax_check_invalid(self):
        """Test syntax checking for invalid code"""
        reward_fn = CodeExecutionReward()
        
        code = """def hello(
    return "world"
"""
        
        is_valid, error = reward_fn.check_syntax(code)
        assert not is_valid
        assert "SyntaxError" in error or error != ""
    
    def test_style_score(self):
        """Test style scoring"""
        reward_fn = CodeExecutionReward()
        
        # Good style code
        good_code = '''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b
'''
        
        good_score = reward_fn.check_style(good_code)
        assert good_score >= 0.7
        
        # Bad style code
        bad_code = '''
def f(x,y):
    z=x+y
    return z
'''
        
        bad_score = reward_fn.check_style(bad_code)
        assert bad_score < 0.7
    
    def test_full_reward_good_code(self):
        """Test full reward computation for good code"""
        reward_fn = CodeExecutionReward(
            test_cases=[(None, "Hello World")]
        )
        
        completion = '''```python
print("Hello World")
```'''
        
        result = reward_fn.compute_reward(completion)
        assert isinstance(result, RewardResult)
        assert 0 <= result.total_score <= 1
        assert result.breakdown["compile"] > 0
    
    def test_full_reward_no_code(self):
        """Test reward when no code found"""
        reward_fn = CodeExecutionReward()
        
        completion = "Here's my solution: just think about it!"
        
        result = reward_fn.compute_reward(completion)
        assert result.total_score == 0.0
        assert "No code found" in result.details.get("error", "")


class TestAgentTrajectoryReward:
    """Test agent trajectory reward"""
    
    def test_successful_trajectory(self):
        """Test reward for successful trajectory"""
        reward_fn = AgentTrajectoryReward()
        
        trajectory = [
            {"role": "assistant", "content": "I'll help you", "tool_calls": [{"name": "search"}]},
            {"role": "tool", "content": "Found result", "success": True},
            {"role": "assistant", "content": "Task completed successfully!"}
        ]
        
        result = reward_fn.compute_reward(trajectory)
        assert result.total_score > 0.5
    
    def test_failed_trajectory(self):
        """Test reward for failed trajectory"""
        reward_fn = AgentTrajectoryReward()
        
        trajectory = [
            {"role": "assistant", "content": "I'll help you", "tool_calls": [{"name": "search"}]},
            {"role": "tool", "content": "Error", "success": False},
            {"role": "assistant", "content": "Sorry, that didn't work."}
        ]
        
        result = reward_fn.compute_reward(trajectory)
        assert result.total_score < 0.5


class TestRewardWeights:
    """Test reward weight configurations"""
    
    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0"""
        with pytest.raises(AssertionError):
            CodeExecutionReward(
                compile_weight=0.5,
                test_weight=0.5,
                style_weight=0.5  # Sum > 1.0
            )
    
    def test_custom_weights(self):
        """Test custom weight configuration"""
        reward_fn = CodeExecutionReward(
            compile_weight=0.4,
            test_weight=0.4,
            style_weight=0.2
        )
        
        assert reward_fn.compile_weight == 0.4
        assert reward_fn.test_weight == 0.4
        assert reward_fn.style_weight == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
