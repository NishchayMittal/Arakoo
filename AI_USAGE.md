# AI Usage Disclosure

## What I Asked AI Tools

1. **Verifiers library research**: I asked AI to research the Prime Intellect `verifiers` library API (`vf.Rubric`, `vf.SingleTurnEnv`, `load_environment()` pattern) to ensure my implementation matches the expected interface.

2. **RL concept explanations**: I asked for explanations of RL concepts (state space, action space, reward hacking, sparse vs. dense rewards) to ensure my documentation was accurate and thorough.

3. **Code structure review**: I asked AI to review the proposed module structure and suggest improvements to the separation of concerns.

## What Code I Accepted from AI Tools

1. **Boilerplate structure**: The initial project structure (`pyproject.toml`, `__init__.py`, test file skeletons) was generated with AI assistance.

2. **Dataset tasks**: The 40 task definitions in `dataset.py` were generated with AI help — I specified the task categories and goal types, and AI helped generate varied instruction phrasings.

3. **Docstrings and comments**: The educational docstrings explaining RL concepts were written with AI assistance.

## What I Modified Myself

1. **Reward function design**: The reward component weights, clipping logic, and the decision to gate `efficiency_reward` on success were my design choices based on understanding reward hacking risks.

2. **State machine design**: The screen graph, element types, and transition logic in `state.py` were designed by me based on the assignment specification.

3. **Safety penalty logic**: The decision to make safety violations binary (0/1) with a high weight (-0.3) rather than proportional was my design choice.

4. **Test cases**: The specific test scenarios (especially edge cases like saving without a draft, tapping labels, toggle oscillation) were designed by me to cover the required test matrix.

5. **Heuristic agent**: The rule-based agent in `run_eval.py` was implemented by me, including the logic for multi-goal task decomposition.

6. **Failure analysis output**: The detailed per-component reward breakdown in the eval output was my addition for debugging insight.

## What I Learned While Completing This Task

1. **Reward engineering is hard**: Designing rewards that don't allow hacking is genuinely difficult. Every reward component I added created a potential exploit that needed mitigation (e.g., efficiency without success gating → immediate finish exploit).

2. **Sparse vs. dense reward tradeoff**: I learned why pure sparse rewards (success only) make learning nearly impossible — the agent needs shaped rewards for intermediate progress, but shaped rewards create hacking opportunities. The balance is crucial.

3. **Verifiers pattern elegance**: The `dataset + rubric + environment` pattern from Prime Intellect is a clean abstraction. By separating reward functions, each can be tested independently and weights can be tuned without changing logic.

4. **Environment design = API design**: A good RL environment is essentially a well-designed API — clear inputs (observations), clear outputs (actions), predictable transitions, and comprehensive error handling.

5. **Train/eval separation matters**: Without it, you can't distinguish between an agent that learned general skills vs. one that memorized specific task solutions. This is especially important for mobile UI tasks where instructions can be paraphrased many ways.
