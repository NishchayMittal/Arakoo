from __future__ import annotations

import argparse
import json
import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mobile_ui_env.env import load_environment, EvalResult
from mobile_ui_env.state import Screen, SCREEN_ELEMENTS


# ─── Heuristic Agent ─────────────────────────────────────────────────────────────

def heuristic_agent(task: dict, observation: str) -> list[dict]:
    """
    A rule-based agent that knows how to solve each task type.(used to set an upper bound)
    """
    goal = task["goal"]
    return _solve_goal(goal)


def _solve_goal(goal: dict) -> list[dict]:
    """Recursively solve a goal (handles multi_goal)."""
    goal_type = goal["type"]
    
    if goal_type == "note_created":
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": goal["title"]},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
    
    elif goal_type == "toggle_enabled":
        toggle_map = {
            "focus_mode": "focus_mode_toggle",
            "notifications": "notifications_toggle",
        }
        target = toggle_map.get(goal["key"], goal["key"] + "_toggle")
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": target},
            {"action": "finish"},
        ]
    
    elif goal_type == "toggle_disabled":
        toggle_map = {
            "focus_mode": "focus_mode_toggle",
            "notifications": "notifications_toggle",
        }
        target = toggle_map.get(goal["key"], goal["key"] + "_toggle")
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": target},
            {"action": "finish"},
        ]
    
    elif goal_type == "info_retrieved":
        screen_buttons = {
            "profile": "profile_button",
            "settings": "settings_button",
            "notes": "notes_button",
        }
        button = screen_buttons.get(goal["screen"], "")
        return [
            {"action": "tap", "target": button},
            {"action": "finish"},
        ]
    
    elif goal_type == "navigate":
        screen_buttons = {
            "notes": "notes_button",
            "settings": "settings_button",
            "profile": "profile_button",
        }
        button = screen_buttons.get(goal["screen"], "")
        return [
            {"action": "tap", "target": button},
            {"action": "finish"},
        ]
    
    elif goal_type == "negative_constraint":
        # Don't do the bad thing — just finish
        return [{"action": "finish"}]
    
    elif goal_type == "multi_goal":
        # Solve sub-goals in sequence, removing intermediate finishes
        all_actions = []
        for sub_goal in goal["goals"]:
            sub_actions = _solve_goal(sub_goal)
            # Remove finish from intermediate solutions
            sub_actions = [a for a in sub_actions if a.get("action") != "finish"]
            # If we need to go back to home for the next sub-goal
            if all_actions and sub_actions:
                all_actions.append({"action": "back"})
            all_actions.extend(sub_actions)
        all_actions.append({"action": "finish"})
        return all_actions
    
    return [{"action": "finish"}]


# ─── Random Agent ────────────────────────────────────────────────────────────────

def random_agent(task: dict, observation: str) -> list[dict]:
    """
    A random agent that picks actions at random.(lower bound since lower succeeding chance)
    """
    max_steps = task.get("max_steps", 8)
    actions = []
    current_screen = Screen.HOME
    
    for _ in range(max_steps):
        action_type = random.choice(["tap", "type", "back", "finish"])
        
        if action_type == "finish":
            actions.append({"action": "finish"})
            break
        elif action_type == "back":
            actions.append({"action": "back"})
            current_screen = Screen.HOME
        elif action_type == "tap":
            elements = list(SCREEN_ELEMENTS[current_screen].keys())
            target = random.choice(elements)
            actions.append({"action": "tap", "target": target})
            # Crude screen tracking
            elem = SCREEN_ELEMENTS[current_screen][target]
            if "navigates_to" in elem:
                current_screen = Screen(elem["navigates_to"])
        elif action_type == "type":
            elements = list(SCREEN_ELEMENTS[current_screen].keys())
            target = random.choice(elements)
            actions.append({"action": "type", "target": target, "text": "random text"})
    
    if not actions or actions[-1].get("action") != "finish":
        actions.append({"action": "finish"})
    
    return actions


# ─── Metrics Computation ─────────────────────────────────────────────────────────

def compute_metrics(results: list[EvalResult]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    n = len(results)
    if n == 0:
        return {}
    
    successes = sum(1 for r in results if r.success)
    total_steps = sum(r.steps_taken for r in results)
    total_invalid = sum(r.invalid_actions for r in results)
    total_actions = sum(r.steps_taken + r.invalid_actions for r in results)
    total_safety = sum(r.safety_violations for r in results)
    
    return {
        "total_tasks": n,
        "success_rate": round(successes / n, 4),
        "average_reward": round(sum(r.reward for r in results) / n, 4),
        "average_steps": round(total_steps / n, 2),
        "invalid_action_rate": round(total_invalid / max(total_actions, 1), 4),
        "safety_violations": total_safety,
    }


def print_metrics(metrics: dict, agent_name: str = "Agent"):
    """Print metrics in a clean, readable format."""
    print(f"\n{'='*60}")
    print(f"  [RESULTS] Evaluation Results -- {agent_name}")
    print(f"{'='*60}")
    print(f"  Total eval tasks:     {metrics.get('total_tasks', 0)}")
    print(f"  Success rate:         {metrics.get('success_rate', 0):.0%}")
    print(f"  Average reward:       {metrics.get('average_reward', 0):.4f}")
    print(f"  Average steps:        {metrics.get('average_steps', 0):.1f}")
    print(f"  Invalid action rate:  {metrics.get('invalid_action_rate', 0):.0%}")
    print(f"  Safety violations:    {metrics.get('safety_violations', 0)}")
    print(f"{'='*60}\n")


def print_failure_analysis(results: list[EvalResult]):
    """Print detailed analysis of failed tasks."""
    failures = [r for r in results if not r.success]
    if not failures:
        print("  [PASS] All tasks passed! No failures to analyze.\n")
        return
    
    print(f"\n{'─'*60}")
    print(f"  [ANALYSIS] Failure Analysis ({len(failures)} failed tasks)")
    print(f"{'─'*60}")
    
    for r in failures:
        print(f"\n  [FAIL] {r.task_id}: {r.instruction}")
        print(f"     Reward: {r.reward:.4f} | Steps: {r.steps_taken} | "
              f"Invalid: {r.invalid_actions} | Safety: {r.safety_violations}")
        if r.reward_breakdown:
            for name, score in r.reward_breakdown.component_scores.items():
                weighted = r.reward_breakdown.component_weighted[name]
                marker = "!!" if weighted < 0 else "  "
                print(f"     {marker} {name}: {score:.4f} (weighted: {weighted:+.4f})")
    print()


def print_per_task_details(results: list[EvalResult]):
    """Print detailed per-task results."""
    print(f"\n{'─'*60}")
    print(f"  [DETAILS] Per-Task Results")
    print(f"{'─'*60}")
    
    for r in results:
        status = "[PASS]" if r.success else "[FAIL]"
        print(f"  {status} {r.task_id}: {r.instruction[:45]:<45s} "
              f"reward={r.reward:.4f}  steps={r.steps_taken}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on the Mobile UI RL Environment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=["heuristic", "random", "both"],
        default="both",
        help="Which agent to evaluate (default: both)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="eval",
        help="Which dataset split to evaluate on (default: eval)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-task details",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load environment
    env = load_environment()
    print(f"\n  >>> Mobile UI Agent RL Environment")
    print(f"  Train tasks: {len(env.dataset)} | Eval tasks: {len(env.eval_dataset)}")
    print(f"  Rubric components: {len(env.rubric.funcs)}")
    print(f"  Evaluating on: {args.split} split\n")
    
    agents = {}
    if args.agent in ("heuristic", "both"):
        agents["Heuristic Baseline"] = heuristic_agent
    if args.agent in ("random", "both"):
        agents["Random Baseline"] = random_agent
    
    for agent_name, agent_fn in agents.items():
        results = env.evaluate_all(agent_fn, split=args.split)
        metrics = compute_metrics(results)
        
        print_metrics(metrics, agent_name)
        
        if args.verbose:
            print_per_task_details(results)
        
        print_failure_analysis(results)


if __name__ == "__main__":
    main()
