"""
mobile_ui_env — RL Environment for Mobile UI Agent Training

This package implements a reinforcement learning environment that simulates
a simple mobile app. An AI agent completes tasks by producing structured
JSON actions and receives rewards based on task success, action validity,
efficiency, and safety.

Built following the Prime Intellect Verifiers pattern.

Quick Start:
    from mobile_ui_env import load_environment
    
    env = load_environment()
    result = env.evaluate(task, agent_actions)
    print(result.reward)
"""

from .env import SingleTurnEnv, load_environment, EvalResult
from .state import AppState, Screen
from .actions import parse_actions, execute_action_sequence, ActionType
from .dataset import build_dataset, get_task_by_id
from .rubric import Rubric, RewardBreakdown

__version__ = "0.1.0"

__all__ = [
    "SingleTurnEnv",
    "load_environment",
    "EvalResult",
    "AppState",
    "Screen",
    "parse_actions",
    "execute_action_sequence",
    "ActionType",
    "build_dataset",
    "get_task_by_id",
    "Rubric",
    "RewardBreakdown",
]
