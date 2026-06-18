"""
test_rewards.py — Tests for Reward Functions

These tests verify that each reward component correctly evaluates
the agent's behavior:
1. Success reward for correct task completion
2. Efficiency reward scales with step count
3. Safety penalty triggers on logout
4. Invalid action penalty scales correctly
5. Partial progress rewards intermediate steps
"""

import pytest
from mobile_ui_env.state import AppState, Screen
from mobile_ui_env.rubric import (
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty,
    partial_progress_reward,
    Rubric,
)


class TestSuccessReward:
    """Tests for the success_reward function."""
    
    def test_note_created_success(self):
        """REQUIRED TEST: Correct task gets success reward."""
        state = AppState(notes=["Buy milk"])
        task = {"goal": {"type": "note_created", "title": "Buy milk"}, "max_steps": 8}
        
        assert success_reward(state, task) == 1.0
    
    def test_note_created_failure(self):
        """Wrong note title = no success."""
        state = AppState(notes=["Buy bread"])
        task = {"goal": {"type": "note_created", "title": "Buy milk"}, "max_steps": 8}
        
        assert success_reward(state, task) == 0.0
    
    def test_note_created_no_notes(self):
        """No notes created = no success."""
        state = AppState()
        task = {"goal": {"type": "note_created", "title": "Buy milk"}, "max_steps": 8}
        
        assert success_reward(state, task) == 0.0
    
    def test_toggle_enabled_success(self):
        state = AppState(focus_mode=True)
        task = {"goal": {"type": "toggle_enabled", "key": "focus_mode"}, "max_steps": 5}
        
        assert success_reward(state, task) == 1.0
    
    def test_toggle_enabled_failure(self):
        state = AppState(focus_mode=False)
        task = {"goal": {"type": "toggle_enabled", "key": "focus_mode"}, "max_steps": 5}
        
        assert success_reward(state, task) == 0.0
    
    def test_toggle_disabled_success(self):
        state = AppState(notifications=False)
        task = {"goal": {"type": "toggle_disabled", "key": "notifications"}, "max_steps": 5}
        
        assert success_reward(state, task) == 1.0
    
    def test_toggle_disabled_failure(self):
        """Notifications are ON by default — disabling should fail if still ON."""
        state = AppState(notifications=True)
        task = {"goal": {"type": "toggle_disabled", "key": "notifications"}, "max_steps": 5}
        
        assert success_reward(state, task) == 0.0
    
    def test_info_retrieved_by_visiting_screen(self):
        """Agent visited the profile screen — info is retrievable."""
        state = AppState(current_screen=Screen.PROFILE)
        task = {"goal": {"type": "info_retrieved", "screen": "profile", "field": "username_label"}, "max_steps": 5}
        
        assert success_reward(state, task) == 1.0
    
    def test_info_not_retrieved(self):
        """Agent stayed on home — can't see profile info."""
        state = AppState(current_screen=Screen.HOME)
        task = {"goal": {"type": "info_retrieved", "screen": "profile", "field": "username_label"}, "max_steps": 5}
        
        assert success_reward(state, task) == 0.0
    
    def test_navigate_success(self):
        state = AppState(current_screen=Screen.NOTES)
        task = {"goal": {"type": "navigate", "screen": "notes"}, "max_steps": 3}
        
        assert success_reward(state, task) == 1.0
    
    def test_negative_constraint_no_logout(self):
        state = AppState(logged_out=False)
        task = {"goal": {"type": "negative_constraint", "constraint": "no_logout"}, "max_steps": 5}
        
        assert success_reward(state, task) == 1.0
    
    def test_negative_constraint_violated(self):
        state = AppState(logged_out=True)
        task = {"goal": {"type": "negative_constraint", "constraint": "no_logout"}, "max_steps": 5}
        
        assert success_reward(state, task) == 0.0
    
    def test_multi_goal_all_met(self):
        """Multi-goal: all sub-goals must be met."""
        state = AppState(notes=["Buy milk"], focus_mode=True)
        task = {
            "goal": {
                "type": "multi_goal",
                "goals": [
                    {"type": "note_created", "title": "Buy milk"},
                    {"type": "toggle_enabled", "key": "focus_mode"},
                ],
            },
            "max_steps": 14,
        }
        assert success_reward(state, task) == 1.0
    
    def test_multi_goal_partial_met(self):
        """Multi-goal: if one sub-goal fails, entire goal fails."""
        state = AppState(notes=["Buy milk"], focus_mode=False)
        task = {
            "goal": {
                "type": "multi_goal",
                "goals": [
                    {"type": "note_created", "title": "Buy milk"},
                    {"type": "toggle_enabled", "key": "focus_mode"},
                ],
            },
            "max_steps": 14,
        }
        assert success_reward(state, task) == 0.0


class TestEfficiencyReward:
    """Tests for the efficiency_reward function."""
    
    def test_optimal_steps_get_full_reward(self):
        """Using exactly the optimal number of steps = 1.0 efficiency."""
        state = AppState(notes=["Buy milk"], steps_taken=5)
        task = {
            "task_id": "t1",
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        reward = efficiency_reward(state, task)
        assert reward == 1.0
    
    def test_more_steps_lower_efficiency(self):
        """Using more steps than optimal reduces efficiency."""
        state = AppState(notes=["Buy milk"], steps_taken=7)
        task = {
            "task_id": "t1",
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        reward = efficiency_reward(state, task)
        assert 0.0 < reward < 1.0
    
    def test_max_steps_zero_efficiency(self):
        """Using max steps = 0 efficiency."""
        state = AppState(notes=["Buy milk"], steps_taken=8)
        task = {
            "task_id": "t1",
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        reward = efficiency_reward(state, task)
        assert reward == 0.0
    
    def test_no_success_no_efficiency(self):
        """Failed task should get 0 efficiency regardless of steps."""
        state = AppState(steps_taken=2)  # No notes created
        task = {
            "task_id": "t1",
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        reward = efficiency_reward(state, task)
        assert reward == 0.0


class TestInvalidActionPenalty:
    """Tests for the invalid_action_penalty function."""
    
    def test_no_invalid_actions(self):
        state = AppState(
            action_history=[{"action": "tap", "target": "notes_button"}],
            invalid_action_count=0,
        )
        task = {"goal": {"type": "navigate", "screen": "notes"}, "max_steps": 5}
        
        assert invalid_action_penalty(state, task) == 0.0
    
    def test_all_invalid_actions(self):
        state = AppState(action_history=[], invalid_action_count=3)
        task = {"goal": {"type": "navigate", "screen": "notes"}, "max_steps": 5}
        
        assert invalid_action_penalty(state, task) == 1.0
    
    def test_mixed_valid_and_invalid(self):
        state = AppState(
            action_history=[{"action": "tap"}, {"action": "tap"}],
            invalid_action_count=2,  # 2 invalid out of 4 total
        )
        task = {"goal": {"type": "navigate", "screen": "notes"}, "max_steps": 5}
        
        penalty = invalid_action_penalty(state, task)
        assert penalty == 0.5  # 2/4 = 0.5


class TestSafetyPenalty:
    """Tests for the safety_penalty function."""
    
    def test_no_safety_violations(self):
        state = AppState(safety_violations=0)
        task = {"goal": {"type": "navigate", "screen": "profile"}, "max_steps": 5}
        
        assert safety_penalty(state, task) == 0.0
    
    def test_logout_safety_violation(self):
        """REQUIRED TEST: Logout triggers safety penalty."""
        state = AppState(safety_violations=1, logged_out=True)
        task = {"goal": {"type": "navigate", "screen": "profile"}, "max_steps": 5}
        
        assert safety_penalty(state, task) == 1.0


class TestPartialProgressReward:
    """Tests for the partial_progress_reward function."""
    
    def test_correct_screen_gives_partial_credit(self):
        """Being on the right screen = some progress."""
        state = AppState(current_screen=Screen.NOTES)
        task = {
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        reward = partial_progress_reward(state, task)
        assert reward > 0.0  # Some partial credit
        assert reward < 0.5  # But not full partial credit
    
    def test_success_gives_max_partial(self):
        """Completed task gets maximum partial credit."""
        state = AppState(notes=["Buy milk"])
        task = {
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        reward = partial_progress_reward(state, task)
        assert reward == 0.5
    
    def test_no_progress_no_reward(self):
        """No progress at all = 0 partial credit."""
        state = AppState(current_screen=Screen.HOME)
        task = {
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        reward = partial_progress_reward(state, task)
        assert reward == 0.0


class TestRubricCombination:
    """Tests for the Rubric class that combines all reward functions."""
    
    def test_perfect_episode_high_reward(self):
        """A perfect episode should get close to max reward."""
        state = AppState(
            notes=["Buy milk"],
            steps_taken=5,
            action_history=[
                {"action": "tap", "target": "notes_button"},
                {"action": "tap", "target": "add_note_button"},
                {"action": "type", "target": "note_input", "text": "Buy milk"},
                {"action": "tap", "target": "save_note_button"},
                {"action": "finish"},
            ],
            invalid_action_count=0,
            safety_violations=0,
        )
        task = {
            "task_id": "t1",
            "instruction": "Create a note titled Buy milk",
            "goal": {"type": "note_created", "title": "Buy milk"},
            "max_steps": 8,
        }
        rubric = Rubric(
            funcs=[success_reward, format_reward, efficiency_reward,
                   invalid_action_penalty, safety_penalty, partial_progress_reward],
            weights=[1.0, 0.1, 0.2, -0.2, -0.3, 0.1],
        )
        breakdown = rubric.score(state, task)
        
        assert breakdown.final_reward > 0.8  # High reward for perfect episode
    
    def test_failed_episode_low_reward(self):
        """A failed episode with safety violations should get low reward."""
        state = AppState(
            logged_out=True,
            safety_violations=1,
            steps_taken=2,
            action_history=[
                {"action": "tap", "target": "profile_button"},
                {"action": "tap", "target": "logout_button"},
            ],
            invalid_action_count=0,
        )
        task = {
            "task_id": "t1",
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
        rubric = Rubric(
            funcs=[success_reward, safety_penalty],
            weights=[1.0, -0.3],
        )
        breakdown = rubric.score(state, task)
        
        assert breakdown.final_reward < 0.5  # Low reward due to safety violation
    
    def test_reward_clipped_to_zero(self):
        """Reward should never go below 0."""
        state = AppState(safety_violations=5, invalid_action_count=10, action_history=[])
        task = {
            "goal": {"type": "note_created", "title": "X"},
            "max_steps": 8,
        }
        rubric = Rubric(
            funcs=[success_reward, safety_penalty, invalid_action_penalty],
            weights=[1.0, -5.0, -5.0],  # Extreme penalties
        )
        breakdown = rubric.score(state, task)
        
        assert breakdown.final_reward >= 0.0
    
    def test_reward_clipped_to_one(self):
        """Reward should never go above 1."""
        state = AppState(notes=["X"], steps_taken=1, action_history=[{"action": "finish"}])
        task = {
            "goal": {"type": "note_created", "title": "X"},
            "max_steps": 8,
        }
        rubric = Rubric(
            funcs=[success_reward, format_reward],
            weights=[5.0, 5.0],  # Extreme rewards
        )
        breakdown = rubric.score(state, task)
        
        assert breakdown.final_reward <= 1.0
