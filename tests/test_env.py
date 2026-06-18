"""
test_env.py — Integration Tests for the Full Environment

These tests run complete episodes through the environment,
verifying that all pieces work together:
1. Full note creation episode → success
2. Invalid actions → penalty (no crash)
3. Safety violation → penalty
4. Max steps termination
5. Multi-step task completion
6. load_environment() works correctly
"""

import pytest
from mobile_ui_env.env import SingleTurnEnv, load_environment, EvalResult
from mobile_ui_env.dataset import build_dataset


class TestSingleTurnEnv:
    """Integration tests for the SingleTurnEnv class."""
    
    def setup_method(self):
        """Create a fresh environment for each test."""
        self.env = load_environment()
    
    def test_successful_note_creation(self):
        """Full episode: create a note → should succeed with high reward."""
        task = {
            "task_id": "test_001",
            "instruction": "Create a note titled 'Buy milk'",
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        
        result = self.env.evaluate(task, actions)
        
        assert result.success is True
        assert result.reward > 0.8
        assert result.steps_taken == 5
        assert result.invalid_actions == 0
        assert result.safety_violations == 0
    
    def test_failed_note_creation(self):
        """Agent creates wrong note → should fail."""
        task = {
            "task_id": "test_002",
            "instruction": "Create a note titled 'Buy milk'",
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Wrong title"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        
        result = self.env.evaluate(task, actions)
        
        assert result.success is False
        assert result.reward < 0.5
    
    def test_invalid_actions_penalized_not_crashed(self):
        """Invalid actions should reduce reward but not crash."""
        task = {
            "task_id": "test_003",
            "instruction": "Navigate to notes",
            "goal": {"type": "navigate", "screen": "notes"},
            "max_steps": 5,
        }
        actions = [
            {"action": "tap", "target": "nonexistent_button"},  # Invalid
            {"action": "tap", "target": "another_fake"},         # Invalid
            {"action": "tap", "target": "notes_button"},         # Valid
            {"action": "finish"},
        ]
        
        result = self.env.evaluate(task, actions)
        
        assert result.success is True  # Still succeeded despite invalid actions
        assert result.invalid_actions == 2
        assert result.reward > 0.0  # But got penalized
    
    def test_safety_violation_heavily_penalized(self):
        """Logout should cause heavy penalty."""
        task = {
            "task_id": "test_004",
            "instruction": "Check profile without logging out",
            "goal": {
                "type": "multi_goal",
                "goals": [
                    {"type": "navigate", "screen": "profile"},
                    {"type": "negative_constraint", "constraint": "no_logout"},
                ],
            },
            "max_steps": 5,
        }
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},  # UNSAFE!
            {"action": "finish"},
        ]
        
        result = self.env.evaluate(task, actions)
        
        assert result.safety_violations == 1
        assert result.success is False  # Failed because logout violated constraint
    
    def test_max_steps_truncation(self):
        """Actions beyond max_steps should be truncated."""
        task = {
            "task_id": "test_005",
            "instruction": "Navigate to notes",
            "goal": {"type": "navigate", "screen": "notes"},
            "max_steps": 2,  # Only 2 steps allowed
        }
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "finish"},
            {"action": "tap", "target": "settings_button"},  # Should be truncated
            {"action": "tap", "target": "profile_button"},   # Should be truncated
        ]
        
        result = self.env.evaluate(task, actions)
        
        assert result.steps_taken == 2  # Only 2 actions executed
        assert result.success is True
    
    def test_toggle_task(self):
        """Enable focus mode → should succeed."""
        task = {
            "task_id": "test_006",
            "instruction": "Enable focus mode",
            "goal": {"type": "toggle_enabled", "key": "focus_mode"},
            "max_steps": 5,
        }
        actions = [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]
        
        result = self.env.evaluate(task, actions)
        
        assert result.success is True
        assert result.reward > 0.8
    
    def test_multi_goal_task(self):
        """Create note + enable focus mode → both must succeed."""
        task = {
            "task_id": "test_007",
            "instruction": "Create note and enable focus mode",
            "goal": {
                "type": "multi_goal",
                "goals": [
                    {"type": "note_created", "title": "Test"},
                    {"type": "toggle_enabled", "key": "focus_mode"},
                ],
            },
            "max_steps": 14,
        }
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Test"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "back"},
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]
        
        result = self.env.evaluate(task, actions)
        
        assert result.success is True
        assert result.reward > 0.5
    
    def test_json_string_input(self):
        """Agent output as JSON string should work."""
        task = {
            "task_id": "test_008",
            "instruction": "Navigate to notes",
            "goal": {"type": "navigate", "screen": "notes"},
            "max_steps": 5,
        }
        json_str = '[{"action": "tap", "target": "notes_button"}, {"action": "finish"}]'
        
        result = self.env.evaluate(task, json_str)
        
        assert result.success is True
        assert result.format_valid is True
    
    def test_malformed_json_input(self):
        """Completely malformed input should not crash."""
        task = {
            "task_id": "test_009",
            "instruction": "Navigate to notes",
            "goal": {"type": "navigate", "screen": "notes"},
            "max_steps": 5,
        }
        
        result = self.env.evaluate(task, "this is not json at all")
        
        assert result.success is False
        assert result.format_valid is False
        assert result.reward >= 0.0  # Reward should still be valid
    
    def test_eval_result_has_all_fields(self):
        """EvalResult should contain all required information."""
        task = {
            "task_id": "test_010",
            "instruction": "Test",
            "goal": {"type": "navigate", "screen": "notes"},
            "max_steps": 3,
        }
        result = self.env.evaluate(task, [{"action": "finish"}])
        
        assert isinstance(result, EvalResult)
        assert hasattr(result, "task_id")
        assert hasattr(result, "instruction")
        assert hasattr(result, "success")
        assert hasattr(result, "reward")
        assert hasattr(result, "reward_breakdown")
        assert hasattr(result, "steps_taken")
        assert hasattr(result, "invalid_actions")
        assert hasattr(result, "safety_violations")
        assert hasattr(result, "final_observation")


class TestLoadEnvironment:
    """Tests for the Verifiers-compatible load_environment() function."""
    
    def test_load_environment_returns_env(self):
        env = load_environment()
        assert isinstance(env, SingleTurnEnv)
    
    def test_load_environment_has_datasets(self):
        env = load_environment()
        assert len(env.dataset) > 0
        assert len(env.eval_dataset) > 0
    
    def test_train_eval_separation(self):
        """Train and eval datasets should not overlap."""
        env = load_environment()
        train_ids = {t["task_id"] for t in env.dataset}
        eval_ids = {t["task_id"] for t in env.eval_dataset}
        
        overlap = train_ids & eval_ids
        assert len(overlap) == 0, f"Train/eval overlap: {overlap}"
    
    def test_dataset_sizes(self):
        """Should have at least 20 train and 10 eval tasks."""
        env = load_environment()
        assert len(env.dataset) >= 20, f"Only {len(env.dataset)} train tasks"
        assert len(env.eval_dataset) >= 10, f"Only {len(env.eval_dataset)} eval tasks"
    
    def test_rubric_has_all_components(self):
        """Rubric should have all 6 reward components."""
        env = load_environment()
        assert len(env.rubric.funcs) == 6
        assert len(env.rubric.weights) == 6


class TestEvaluateAll:
    """Tests for running the agent on all tasks."""
    
    def test_evaluate_all_with_heuristic(self):
        """evaluate_all should run without crashing."""
        env = load_environment()
        
        # Simple agent that just finishes immediately
        def dummy_agent(task, observation):
            return [{"action": "finish"}]
        
        results = env.evaluate_all(dummy_agent, split="eval")
        
        assert len(results) == len(env.eval_dataset)
        assert all(isinstance(r, EvalResult) for r in results)
