from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from .state import AppState, Screen
from .dataset import get_optimal_steps


# ─── Individual Reward Functions ─────────────────────────────────────────────────
# Each function takes the final state and task info, returns a float.
# Positive values = reward, negative values = penalty.


def success_reward(state: AppState, task: dict, **kwargs) -> float:
    """
    SPARSE REWARD: 1.0 if the goal is fully completed, 0.0 otherwise.
    """
    goal = task["goal"]
    return _check_goal(state, goal)


def _check_goal(state: AppState, goal: dict) -> float:
    """Recursively check if a goal (or multi-goal) is satisfied."""
    goal_type = goal["type"]
    
    if goal_type == "note_created":
        # Check if a note with the exact title exists
        return 1.0 if goal["title"] in state.notes else 0.0
    
    elif goal_type == "toggle_enabled":
        # Check if the specified toggle is ON
        return 1.0 if getattr(state, goal["key"], False) else 0.0
    
    elif goal_type == "toggle_disabled":
        # Check if the specified toggle is OFF
        return 1.0 if not getattr(state, goal["key"], True) else 0.0
    
    elif goal_type == "info_retrieved":
        # Agent must have navigated to the correct screen (info is visible there)
        target_screen = Screen(goal["screen"])
        # Check if agent visited this screen at any point (check action history)
        visited = _has_visited_screen(state, target_screen)
        return 1.0 if visited else 0.0
    
    elif goal_type == "navigate":
        # Agent must be on or have visited the target screen
        target_screen = Screen(goal["screen"])
        visited = _has_visited_screen(state, target_screen)
        return 1.0 if visited else 0.0
    
    elif goal_type == "negative_constraint":
        # Agent must NOT have done something dangerous
        if goal["constraint"] == "no_logout":
            return 1.0 if not state.logged_out else 0.0
        return 1.0  # Unknown constraint — assume satisfied
    
    elif goal_type == "multi_goal":
        # ALL sub-goals must be satisfied
        sub_scores = [_check_goal(state, g) for g in goal["goals"]]
        if not sub_scores:
            return 0.0
        return 1.0 if all(s == 1.0 for s in sub_scores) else 0.0
    
    return 0.0  # Unknown goal type


def _has_visited_screen(state: AppState, target_screen: Screen) -> bool:
    """Check if the agent has visited a specific screen during the episode."""
    # Check current screen
    if state.current_screen == target_screen:
        return True
    # Check action history for navigation to that screen
    screen_buttons = {
        Screen.NOTES: "notes_button",
        Screen.SETTINGS: "settings_button",
        Screen.PROFILE: "profile_button",
    }
    target_button = screen_buttons.get(target_screen)
    if target_button:
        for action in state.action_history:
            if action.get("action") == "tap" and action.get("target") == target_button:
                return True
    return False


def format_reward(state: AppState, task: dict, **kwargs) -> float:
    """
    DENSE REWARD: Rewards valid JSON/action format.
    Returns 1.0 if all actions were valid format, scaled down for each invalid one.
    """
    total_actions = len(state.action_history)
    if total_actions == 0:
        return 0.0  # No actions = no format to judge
    
    # Count actions that had valid format (not counting semantic validity,
    # just structural validity — correct fields present)
    valid_count = total_actions  # action_history only contains parsed actions
    # The invalid_action_count includes both format and semantic errors
    # but for format_reward we're more lenient
    
    return 1.0  # If we got here, the format was parseable


def efficiency_reward(state: AppState, task: dict, **kwargs) -> float:
    """
    DENSE/SHAPED REWARD: Rewards completing the task in fewer steps.
    
    Formula: max(0, 1 - (actual_steps - optimal_steps) / max_steps)
    """
    # Only reward efficiency if the task was completed
    task_success = success_reward(state, task)
    if task_success < 1.0:
        return 0.0
    
    actual_steps = state.steps_taken
    optimal_steps = get_optimal_steps(task)
    max_steps = task.get("max_steps", 8)
    
    if max_steps <= optimal_steps:
        return 1.0 if actual_steps <= optimal_steps else 0.0
    
    efficiency = max(0.0, 1.0 - (actual_steps - optimal_steps) / (max_steps - optimal_steps))
    return round(efficiency, 4)


def invalid_action_penalty(state: AppState, task: dict, **kwargs) -> float:
    """
    PENALTY: Penalizes the fraction of actions that were invalid.
   Returns a value between 0.0 (no invalid actions) and 1.0 (all actions were invalid).
    Note: This is SUBTRACTED from the final reward, so higher = worse.
    """
    total_actions = len(state.action_history) + state.invalid_action_count
    if total_actions == 0:
        return 0.0
    
    return round(state.invalid_action_count / total_actions, 4)


def safety_penalty(state: AppState, task: dict, **kwargs) -> float:
    """
    PENALTY: Heavy penalty for unsafe actions (like logging out).
Returns 1.0 if any safety violation occurred, 0.0 otherwise.
    """
    return 1.0 if state.safety_violations > 0 else 0.0


def partial_progress_reward(state: AppState, task: dict, **kwargs) -> float:
    """
    SHAPED REWARD: Gives partial credit for making progress toward the goal.
    Returns:
        0.0 to 0.5 based on intermediate progress.
    """
    goal = task["goal"]
    
    # Already succeeded — no need for partial credit
    if success_reward(state, task) == 1.0:
        return 0.5  # Max partial reward
    
    return _check_partial_progress(state, goal)


def _check_partial_progress(state: AppState, goal: dict) -> float:
    """Calculate partial progress toward a goal."""
    goal_type = goal["type"]
    
    if goal_type == "note_created":
        # +0.15 for being on notes screen
        # +0.25 for having a draft started
        # +0.35 for having the correct text typed
        progress = 0.0
        if state.current_screen == Screen.NOTES or _has_visited_screen(state, Screen.NOTES):
            progress += 0.15
        if state.current_note_draft is not None:
            progress += 0.10
            if state.current_note_draft == goal["title"]:
                progress += 0.10
        return progress
    
    elif goal_type in ("toggle_enabled", "toggle_disabled"):
        # +0.2 for being on settings screen
        if state.current_screen == Screen.SETTINGS or _has_visited_screen(state, Screen.SETTINGS):
            return 0.2
        return 0.0
    
    elif goal_type in ("info_retrieved", "navigate"):
        target = Screen(goal.get("screen", "home"))
        if _has_visited_screen(state, target):
            return 0.25
        return 0.0
    
    elif goal_type == "multi_goal":
        # Average partial progress of sub-goals
        sub_scores = []
        for g in goal["goals"]:
            if g["type"] == "negative_constraint":
                sub_scores.append(_check_goal(state, g) * 0.5)
            else:
                sub_scores.append(_check_partial_progress(state, g))
        return sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
    
    return 0.0


# ─── Rubric Class (Verifiers-Compatible) ─────────────────────────────────────────

@dataclass
class RewardBreakdown:
    """Detailed breakdown of how the final reward was calculated."""
    component_scores: dict[str, float] = field(default_factory=dict)
    component_weighted: dict[str, float] = field(default_factory=dict)
    final_reward: float = 0.0

    def __str__(self) -> str:
        lines = ["Reward Breakdown:"]
        for name, raw in self.component_scores.items():
            weighted = self.component_weighted[name]
            lines.append(f"  {name:30s} raw={raw:+.4f}  weighted={weighted:+.4f}")
        lines.append(f"  {'─' * 60}")
        lines.append(f"  {'FINAL (clipped to [0,1])':30s}          = {self.final_reward:.4f}")
        return "\n".join(lines)


class Rubric:
    """
    Combines multiple reward functions into one final score.
    
    This mirrors the Prime Intellect vf.Rubric interface:
    - funcs: list of reward functions
    - weights: how much each function contributes to the total
    
    Some functions are REWARDS (positive contributions) and some are
    PENALTIES (negative contributions). The weights and signs determine
    which is which:
    - success_reward (weight +1.0) — adds to the score
    - safety_penalty (weight -0.3) — subtracts from the score
    """
    
    def __init__(
        self,
        funcs: list[Callable],
        weights: list[float] | None = None,
    ):
        self.funcs = funcs
        self.weights = weights or [1.0] * len(funcs)
        
        if len(self.funcs) != len(self.weights):
            raise ValueError(
                f"Number of functions ({len(self.funcs)}) must match "
                f"number of weights ({len(self.weights)})"
            )
    
    def score(self, state: AppState, task: dict, **kwargs) -> RewardBreakdown:
        """
        Compute the final reward by running all reward functions and combining them.
        
        The formula is:
            final = sum(weight_i * func_i(state, task))
            final = clip(final, 0, 1)
        
        Clipping to [0, 1] ensures the reward is always in a reasonable range
        for RL training.
        """
        breakdown = RewardBreakdown()
        
        total = 0.0
        for func, weight in zip(self.funcs, self.weights):
            raw_score = func(state, task, **kwargs)
            weighted_score = weight * raw_score
            
            func_name = func.__name__
            breakdown.component_scores[func_name] = raw_score
            breakdown.component_weighted[func_name] = weighted_score
            
            total += weighted_score
        
        # Clip to [0, 1] range
        breakdown.final_reward = round(max(0.0, min(1.0, total)), 4)
        
        return breakdown
