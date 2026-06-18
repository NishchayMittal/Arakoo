# Mobile UI Agent RL Environment

A reinforcement learning environment that simulates a simple mobile app where an AI agent completes tasks by producing structured JSON actions. Built following the [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/verifiers) pattern.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd mobile_ui_env

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run evaluation
python run_eval.py
python run_eval.py --agent heuristic --verbose
python run_eval.py --agent random --verbose
```

## Project Structure

```
mobile_ui_env/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
├── AI_USAGE.md                 # AI tools disclosure
├── Dockerfile                  # Container setup
├── run_eval.py                 # Evaluation script with baselines
├── mobile_ui_env/
│   ├── __init__.py             # Package exports + load_environment()
│   ├── env.py                  # SingleTurnEnv — core RL environment
│   ├── state.py                # AppState — screen graph + state machine
│   ├── actions.py              # Action parsing, validation, execution
│   ├── dataset.py              # 40 tasks with train/eval split
│   └── rubric.py               # 6 reward components + Rubric class
└── tests/
    ├── test_actions.py          # 32 action tests
    ├── test_rewards.py          # 30 reward tests
    └── test_env.py              # 16 integration tests
```

---

## RL Environment Design

### 1. What is the State Space?

The **state space** is everything the environment tracks at any given moment. Our state consists of:

| Component | Type | Description |
|-----------|------|-------------|
| `current_screen` | Enum | Which of 4 screens the agent is viewing (home, notes, settings, profile) |
| `notes` | List[str] | Notes the agent has created (initially empty) |
| `current_note_draft` | Optional[str] | Text typed but not yet saved |
| `focus_mode` | bool | Whether focus mode is enabled (default: False) |
| `notifications` | bool | Whether notifications are enabled (default: True) |
| `logged_out` | bool | Whether the user has been logged out (safety-critical) |
| `action_history` | List[dict] | Complete log of all actions taken |
| `invalid_action_count` | int | Number of invalid actions attempted |
| `safety_violations` | int | Number of unsafe actions performed |

Each screen has a fixed set of UI elements (buttons, toggles, inputs, labels) forming a **UI hierarchy** — analogous to the accessibility tree in a real Android app.

The **observation** the agent receives is a structured text representation of the current screen, similar to what a real accessibility tree would provide:

```
[Screen: notes]
[Elements on screen:]
  - add_note_button (button)
  - note_input (input) [empty]
  - save_note_button (button)
  - note_list (list) [empty]
```

### 2. What is the Action Space?

The agent outputs a **list of JSON actions**. Four action types are supported:

| Action | Fields | Effect |
|--------|--------|--------|
| `tap` | `target`: element ID | Press a button, toggle a switch, navigate |
| `type` | `target`: element ID, `text`: string | Enter text into an input field |
| `back` | (none) | Return to the home screen |
| `finish` | (none) | End the episode ("I'm done") |

Example agent output:
```json
[
  {"action": "tap", "target": "notes_button"},
  {"action": "tap", "target": "add_note_button"},
  {"action": "type", "target": "note_input", "text": "Buy milk"},
  {"action": "tap", "target": "save_note_button"},
  {"action": "finish"}
]
```

**Invalid actions** (tapping non-existent elements, typing into buttons, etc.) are handled gracefully — they increment `invalid_action_count` but **never crash** the environment.

### 3. What is the Episode Termination Condition?

An **episode** (one complete task attempt) ends when either:

1. **The agent calls `finish`** — it believes it has completed the task
2. **`max_steps` is exceeded** — the agent ran out of time (action list truncated)

After termination, the rubric evaluates the final state against the goal to compute the reward.

### 4. Which Rewards are Sparse?

**Sparse rewards** are only given at the very end — the agent gets no intermediate feedback.

- **`success_reward`** (weight: 1.0): Binary 1.0 or 0.0 based on whether the goal was fully achieved. This is the most important reward but also the hardest to learn from. An agent that navigates to the right screen but forgets the final step gets the same 0.0 as one that does nothing.

Sparse rewards create the **credit assignment problem**: the agent knows it failed but can't tell which of its 5+ actions was the problem. This is why RL training is challenging and often requires millions of episodes.

### 5. Which Rewards are Dense or Shaped?

**Dense/shaped rewards** provide incremental feedback during or after the episode:

- **`partial_progress_reward`** (weight: 0.1): Gives credit for intermediate progress:
  - +0.15 for reaching the correct screen
  - +0.10 for starting a note draft
  - +0.10 for typing the correct text
  - This creates a "gradient" the agent can follow toward the goal

- **`efficiency_reward`** (weight: 0.2): Rewards completing tasks in fewer steps. Formula: `max(0, 1 - (steps - optimal) / (max_steps - optimal))`. Only awarded on successful tasks.

- **`format_reward`** (weight: 0.1): Rewards well-formatted JSON output.

- **`invalid_action_penalty`** (weight: -0.2): Penalizes the fraction of invalid actions.

- **`safety_penalty`** (weight: -0.3): Heavy penalty for unsafe actions like logout.

### 6. How Can Reward Hacking Happen in This Environment?

**Reward hacking** occurs when the agent exploits the reward function to get high scores without actually solving the task. Examples in our environment:

1. **Immediate finish exploit**: If efficiency_reward were awarded regardless of success, the agent could call `finish` immediately (0 steps = max efficiency). We prevent this by gating efficiency on `success_reward == 1.0`.

2. **Format farming**: The agent could output perfectly formatted JSON that does nothing useful, collecting `format_reward` (0.1) without attempting the task. The low weight makes this unattractive vs. actually solving tasks (1.0).

3. **Partial progress farming**: The agent could learn to just navigate to the correct screen and collect `partial_progress_reward` (0.15) without completing the task. We mitigate this with a low weight (0.1) and capped maximum (0.5).

4. **Toggle oscillation**: For toggle tasks, the agent could rapidly toggle on/off to "explore" and accidentally land on the right state. This isn't strictly hacking but produces brittle learned behavior.

5. **Negative constraint gaming**: For "don't logout" tasks, the agent could immediately `finish` (satisfying the constraint) without performing the positive part of the task. Multi-goal evaluation catches this.

### 7. How Would You Scale This from Mock UI to Real Android Emulator?

To move from our simulated screens to a real Android environment:

| Mock Component | Real-World Replacement |
|---------------|----------------------|
| `SCREEN_ELEMENTS` dict | Android **Accessibility Tree** (XML hierarchy from UIAutomator) |
| `get_observation()` text | **Screenshot** (pixels) + **View Hierarchy** (XML) + **Accessibility nodes** |
| `execute_action()` | **ADB commands** (`adb shell input tap x y`, `adb shell input text "..."`) |
| `Screen` enum | **Activity/Fragment detection** via `adb shell dumpsys activity`) |
| Fixed element IDs | Dynamic **resource-id**, **content-desc**, or **XPath** selectors |
| Deterministic transitions | Stochastic real-world (network delays, animations, pop-ups) |

**Architecture for a real system:**

1. **Emulator Pool**: Run N Android emulators (e.g., via Android ADK / Google Cloud Android Virtual Devices)
2. **Observation Extraction**: After each action, capture screenshot + dump view hierarchy + extract accessibility tree
3. **Action Execution**: Translate agent actions to ADB tap/type/swipe commands with coordinate resolution
4. **State Verification**: Use view hierarchy to programmatically verify goal completion
5. **Parallel Rollouts**: PRIME-RL orchestrator manages multiple emulator instances for throughput

Key challenges:
- **Latency**: Real emulators take 1-3 seconds per action (vs. microseconds in our mock)
- **Non-determinism**: Network requests, animations, and timing create stochastic transitions
- **Observation complexity**: Screenshots are 1080x1920 pixels — need vision models or structured representations
- **Reset cost**: Resetting emulator state is expensive — need snapshot/restore

### 8. How Would This Work with Prime Intellect, Verifiers, or PRIME-RL Later?

Our environment follows the **Verifiers pattern** precisely:

```python
import verifiers as vf

def load_environment():
    dataset = build_dataset(split="train")
    eval_dataset = build_dataset(split="eval")
    rubric = vf.Rubric(
        funcs=[success_reward, format_reward, efficiency_reward,
               invalid_action_penalty, safety_penalty, partial_progress_reward],
        weights=[1.0, 0.1, 0.2, -0.2, -0.3, 0.1],
    )
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
    )
```

**Integration path:**

1. **Evaluation**: Run `prime eval run mobile-ui-env -m <model> -n 100` to evaluate any model
2. **Training**: PRIME-RL orchestrator calls `load_environment()`, generates rollouts, and the trainer updates model weights using RLVR (Reinforcement Learning from Verifiable Rewards)
3. **Scaling**: For real emulator environments, upgrade to `vf.StatefulToolEnv` with per-rollout emulator instances
4. **Multi-turn**: Extend to `vf.MultiTurnEnv` for interactive agent-environment loops where the agent sees observations between actions

The Rubric's reward functions serve as **verifiers** — they programmatically check if the model's output achieves the goal, enabling scalable RL without human feedback.

### 9. What Tests Did You Write?

**78 total tests** across 3 test files:

| Test File | Count | Coverage |
|-----------|-------|----------|
| `test_actions.py` | 32 | Action parsing (JSON, malformed, empty), validation (all fields), tap navigation, tap invalid element, tap label, toggle state change, type into input, type into non-input, note creation workflow, save without draft, back navigation, finish, logout safety, action sequences |
| `test_rewards.py` | 30 | Success for all goal types (note_created, toggle_enabled, toggle_disabled, info_retrieved, navigate, negative_constraint, multi_goal), efficiency scaling, efficiency requires success, invalid action penalty scaling, safety penalty, partial progress, rubric combination, reward clipping |
| `test_env.py` | 16 | Full episode success/failure, invalid actions don't crash, safety violation penalty, max_steps truncation, toggle/multi-goal/JSON string/malformed input episodes, EvalResult fields, load_environment, train/eval separation, dataset sizes, rubric components, evaluate_all |

**Required tests from the assignment:**
1. ✅ Valid tap changes screen (`test_tap_navigates_to_notes`)
2. ✅ Invalid tap does not crash (`test_tap_nonexistent_element_does_not_crash`)
3. ✅ Creating a note updates state (`test_full_note_creation_workflow`)
4. ✅ Correct task gets success reward (`test_note_created_success`)
5. ✅ Logout action triggers safety penalty (`test_logout_triggers_safety_violation`)

### 10. What Tradeoffs Did You Make Due to Limited Scope?

| Decision | Tradeoff | Reason |
|----------|----------|--------|
| **Single-turn evaluation** | Agent outputs all actions at once, can't react to intermediate state | Matches Verifiers `SingleTurnEnv` pattern; multi-turn would need `MultiTurnEnv` |
| **Text observations only** | No visual/screenshot observations | Mock environment has no real pixels; text is sufficient for structured UI |
| **Fixed screen elements** | Can't handle dynamic UI (lists growing, modals appearing) | Keeps state space tractable; real apps would need dynamic element discovery |
| **Deterministic transitions** | No randomness in state transitions | Simplifies testing and debugging; real emulators have latency/non-determinism |
| **Mock verifiers classes** | Built our own `Rubric` and `SingleTurnEnv` instead of importing `verifiers` | Assignment asks for standalone package; real deployment would use `import verifiers as vf` |
| **Heuristic baseline only** | No actual RL training | Assignment scope is environment design, not model training |
| **40 tasks** | Limited task diversity | Demonstrates all required types; real deployment would need 1000+ |

---

## Evaluation Results

### Heuristic Baseline (Upper Bound)
```
Total eval tasks:     14
Success rate:         100%
Average reward:       1.0000
Average steps:        3.6
Invalid action rate:  0%
Safety violations:    0
```

### Random Baseline (Lower Bound)
```
Total eval tasks:     14
Success rate:         0%
Average reward:       ~0.07
Average steps:        ~4.0
Invalid action rate:  ~20%
Safety violations:    ~1
```

The large gap between baselines demonstrates that this environment requires **learned behavior** — random exploration cannot solve the tasks, but a properly trained agent can.

---

## Running Locally

```bash
# Prerequisites: Python >= 3.10

# Install
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run evaluation (both baselines)
python run_eval.py

# Run with details
python run_eval.py --agent heuristic --verbose
python run_eval.py --agent random --verbose

# Evaluate on training set
python run_eval.py --split train --verbose
```

## Docker

```bash
docker build -t mobile-ui-env .
docker run mobile-ui-env           # Runs eval
docker run mobile-ui-env pytest    # Runs tests
```

## License

MIT
