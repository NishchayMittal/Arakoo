from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import AppState
from .actions import parse_actions, execute_action_sequence
from .rubric import (
    Rubric,
    RewardBreakdown,
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty,
    partial_progress_reward,
)
from .dataset import build_dataset


@dataclass
class EvalResult:
    task_id: str
    instruction: str
    success: bool
    reward: float
    reward_breakdown: RewardBreakdown
    steps_taken: int
    invalid_actions: int
    safety_violations: int
    format_valid: bool
    action_log: list[dict]
    final_observation: str


class SingleTurnEnv:
    
    def __init__(
        self,
        dataset: list[dict],
        eval_dataset: list[dict] | None = None,
        rubric: Rubric | None = None,
    ):
        self.dataset = dataset
        self.eval_dataset = eval_dataset or []
        self.rubric = rubric or self._default_rubric()
    
    @staticmethod
    def _default_rubric() -> Rubric:
        """Create the default rubric with all reward components."""
        return Rubric(
            funcs=[
                success_reward,     
                format_reward,        
                efficiency_reward,   
                invalid_action_penalty,  
                safety_penalty,       
                partial_progress_reward, 
            ],
            weights=[1.0, 0.1, 0.2, -0.2, -0.3, 0.1],
            #main reward for task completion
            #bonus for efficiency
            #penalty for invalid actions
            #STRONG penalty for unsafe actions
            #small bonus for progress
            
            
            
            

        )
    
    def reset(self, task: dict) -> tuple[AppState, str]:
        """
        Start a new episode for the given task.
        Returns:
            (state, observation): The fresh state and initial observation text.
        """
        state = AppState()
        observation = state.get_observation()
        return state, observation
    
    def step(
        self,
        state: AppState,
        raw_actions: str | list,
        task: dict,
    ) -> EvalResult:
        # Parse actions from agent output
        actions, format_valid = parse_actions(raw_actions)
        
        # Enforce max_steps by truncating if necessary
        max_steps = task.get("max_steps", 8)
        if len(actions) > max_steps:
            actions = actions[:max_steps]
        
        # Execute all actions
        execute_action_sequence(state, actions)
        
        # If agent didn't call finish, mark episode as done anyway
        if not state.episode_finished:
            state.episode_finished = True
        
        # Handle format failure — if parsing failed entirely, penalize
        if not format_valid:
            state.invalid_action_count += 1
        
        # Compute reward using the rubric
        reward_breakdown = self.rubric.score(state, task)
        
        # Determine success
        success = success_reward(state, task) == 1.0
        
        return EvalResult(
            task_id=task["task_id"],
            instruction=task["instruction"],
            success=success,
            reward=reward_breakdown.final_reward,
            reward_breakdown=reward_breakdown,
            steps_taken=state.steps_taken,
            invalid_actions=state.invalid_action_count,
            safety_violations=state.safety_violations,
            format_valid=format_valid,
            action_log=state.action_history,
            final_observation=state.get_observation(),
        )
    
    def evaluate(self, task: dict, raw_actions: str | list) -> EvalResult:
        """
        Convenience method: reset + step in one call.
        """
        state, _ = self.reset(task)
        return self.step(state, raw_actions, task)
    
    def evaluate_all(
        self,
        agent_fn: Any,
        split: str = "eval",
    ) -> list[EvalResult]:
        """
        Run the agent on all tasks in a dataset split.
        
        Args:
            agent_fn: A callable that takes (task, observation) and returns actions.
                      Can be a heuristic, random agent, or LLM call.
            split: Which dataset to use ("train" or "eval").
        
        Returns:
            List of EvalResult for each task.
        """
        dataset = self.eval_dataset if split == "eval" else self.dataset
        results = []
        
        for task in dataset:
            state, observation = self.reset(task)
            actions = agent_fn(task, observation)
            result = self.step(state, actions, task)
            results.append(result)
        
        return results


# ─── Environment Loader (Verifiers-Compatible) ───────────────────────────────────

def load_environment() -> SingleTurnEnv:
    """
    Prime Intellect Verifiers-compatible entry point.
    
    This is the standard function that the Verifiers framework calls to
    load your environment. It returns a fully configured SingleTurnEnv
    with dataset, eval dataset, and rubric.
    
    Usage:
        import mobile_ui_env
        env = mobile_ui_env.load_environment()
        result = env.evaluate(task, agent_actions)
    """
    dataset = build_dataset(split="train")
    eval_dataset = build_dataset(split="eval")
    
    rubric = Rubric(
        funcs=[
            success_reward,
            format_reward,
            efficiency_reward,
            invalid_action_penalty,
            safety_penalty,
            partial_progress_reward,
        ],
        weights=[1.0, 0.1, 0.2, -0.2, -0.3, 0.1],
    )
    
    return SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
    )
