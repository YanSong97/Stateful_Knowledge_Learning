
ACT_PLAN_SYSTEM_PROMPT="""\
You are an agent solving the FrozenLake game. You need to avoid falling into the hole and arrive at the goal position.
The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available movements are:
Left, Down, Right, Up

At each game state, you will receive an observation that tells you the status of the map and your location.
At each game state, you can interact with an imaginary simulator to simulate a few steps ahead to determine the optimal action at current game state. Make sure the action is reasonable while interacting with the simulator as less as possible.

[Simulation Rule]
During simulation, you can choose one from the following decisions:
1. MOVE: output "<move>[your move]</move>" to move in the corresponding direction in the simulator (e.g. Left, Down, Right, Up), you can choose this if you want to try out a particular movement, you will receive feedback from the simulator after executing this movement;
2. RESET: output"<reset>1</reset>" to reset simulator to the original game state, you can choose this if you feel that current simulation trajectory is wrong and would like to re-plan the simulation;
3. END: output "<end>1</end>" to end the simulation, you can choose this if you feel that the interaction history is sufficient to conclude an optimal movement at current game state.
You will also be given available decisions at each simulation steps. Only output decision that is available in the query.
At the end of each simulation, you will be asked to rethink your interaction history in general and summerize a final movement decision for current game state (at simulation step 1). You are required to generate "<answer>[you move]</answer>" for the current game state.

[Simulation Examples]
Example answer format to move: <think>To forbid the hole and go to the target, I should try go left then go up.</think><move>Left</move>
Example answer format to reset <think>The current trajectory show less promise, I should reset to original state and replan.</think><reset>1</reset>
Example answer format to end: <think>From the interaction history, we think we can conclude with an optimal action at current game state</think><end>1</end>
Example answer format to summerize: <think>From simulation 1, we known that move down can lead to falling into holes so we should avoid this movement......</think> <answer>Up</answer>

Strictly follow the required format or otherwise you will be penalized.
"""


ACT_QUERY_PROMPT="""\
Game Step {turn_idx} with {turn_left} steps budget left acting in the game:
Game State:
{state}
"""

ACT_QUERY_PROMPT_WITH_INFO="""\
Game Step {turn_idx} with {turn_left} steps budget left acting in the game:
Game State:
{state}
"""

PLAN_QUERY_PROMPT="""\
Interaction turn {turn_idx}, step {step_idx}, {extra_info}:
Now the state of simulator is:
{state}
Available decisions:
{available_move}
You have {turn_left} budget left for reset the simulation. You can now plan for maximum {max_step} number of steps ahead. Strictly follow the required format.
"""

SUMMARIZE_PLAN_QUERY_PROMPT="""\
The interaction ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moving direction at simulation step 1. Output your final decision for the root game-state (at simulation step 1):
{state}
Always output: <think>[Your thoughts]</think><answer>[your move]</answer> with no extra text. Strictly follow the required format.
"""

PLAN_SUCCESS_PROMPT="""\
The final board position of this simulation:
{state}

The simulation ends. You have successfully solve the game within this simulation! Congratulation!
"""

PLAN_FAIL_PROMPT="""\
The final board position of this simulation:
{state}
{extra_info}
The simulation ends. You did not solve the game within this simulation {fail_reason}.
"""


SUCCESS_PROMPT = """\
The final board position:
{state}

The game ends. You have successfully solve the game! Congradulations!
"""

FAIL_PROMPT = """\
The final board position:
{state}

The game ends. You did not solve the game due to {fail_reason}.
"""
NO_DECISION_DETECTED="""\
No valid decision detected. End current simulation turn.
"""

INDEPENDENT_SUMMARY_INSTRUCTION="""\
You are a professional game player playing FrozenLake. 

# Instruction
Your task is to analysize the simulation history and summarize the best move at root state.
"""

INDEPENDENT_SUMMARY_PROMPT="""\
[Summarizing stage]

The simulation ends. Lets pause, check the simulation history, and summarize from the simulation history to get the best move at root state.

# Root State

{state}

# Simulation History

{history}

## Output format
<think> Your summary </think> <answer>Your Move at root state</answer>
"""



QWEN_PLAN_ACT_PROMPT_SET = {
    "system_prompt": ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": ACT_QUERY_PROMPT,
    "act_query_prompt_with_info": ACT_QUERY_PROMPT_WITH_INFO,
    "plan_query_prompt": PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_info": PLAN_QUERY_PROMPT,
    "plan_action_query_prompt": SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
    "independent_summary_instruction": INDEPENDENT_SUMMARY_INSTRUCTION,
    "independent_summary_prompt": INDEPENDENT_SUMMARY_PROMPT,
}

# ====================== FrozenLake Direct Inference======================
FROZEN_LAKE_POLICY_SIMPLE_PROMPT_RAGEN = """\
You are solving the FrozenLake puzzle. Forbid the hole and go to the goal.
Example answer format: <think>To forbid the hole and go to the goal, I should go left then go up.</think><answer>Left</answer>

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to only 1 actions.
"""

FROZEN_LAKE_POLICY_SIMPLE_PROMPT_RAGEN_WITH_HINT_SLIPPERY = """\
You are solving the FrozenLake puzzle. Forbid the hole and go to the goal.
The lake is slippery  so the player may move perpendicular to the intended direction sometimes. Make sure to consider this when planning your action.

Example answer format: <think>To forbid the hole and go to the goal, I should go left then go up.</think><answer>Left</answer>

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to only 1 actions.
"""

REFLEXION_PROMPT = """You will be given the history of a past experience in which you were trying to solve a frozenlake map but failed. \
Do not summarize your environment, but rather think about the strategy and path you took to attempt to solve the map. \
Devise a concise, new plan of actions that accounts for your mistake with reference to specific actions that you should have taken. \
For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. \
You will need this later when you are solving the same map again. Note: the slippery probability does not neccessarily follow 1/3 for each action, figure out the actual probability from the history by yourself.
"""

KNOWLEDGE_UPDATE_PROMPT = """You will be given the history of a past experience starting at the same root state, in which you were trying to solve a frozenlake map but failed. 
You will also be given your previous understanding of the root state.

Based on the history, update your understanding of the root state. Note: the slippery probability does not neccessarily follow 1/3 for each action, figure out the actual probability from the history by yourself.
"""

ADVANCED_KNOWLEDGE_UPDATE_PROMPT="""You will be given the history of a past experience starting at the same root state, in which you were trying to solve a frozenlake map but failed. 
You will also be given your previous understanding of the root state.

Based on the history, and your previous understanding, try to reason the reason for your failure (e.g. slippery dynamics, your poor spatial recognition of position, etc.), and correspondingly update your understanding of the root state.
"""

KNOWLEDGE_UPDATE_PROMPT_TD = """You will be given the history of a past experience starting at the same root state, in which you were trying to solve a frozenlake map but failed. 
You might not see the final outcome in each trajectory, but you will see the understanding of the final state in each trajectory. Take that into your consideration when deciding the subsequent quality of each action.

You will also be given your previous understanding of the root state. Based on the history, update your understanding of the root state.
"""



FROZEN_LAKE_STATE_USER_PROMPT_RAGEN = """\
Turn {turn_idx}:
State:
{state}
You have {turn_left} turns left. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.
"""


RL_PROMPT_SET = {
    "system_prompt": FROZEN_LAKE_POLICY_SIMPLE_PROMPT_RAGEN,
    "query_prompt": FROZEN_LAKE_STATE_USER_PROMPT_RAGEN,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "reflexion_prompt": REFLEXION_PROMPT
}

RL_PROMPT_SET_HINT_SLIPPERY={
    "system_prompt": FROZEN_LAKE_POLICY_SIMPLE_PROMPT_RAGEN_WITH_HINT_SLIPPERY,
    "query_prompt": FROZEN_LAKE_STATE_USER_PROMPT_RAGEN,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "reflexion_prompt": REFLEXION_PROMPT
}



# ============== Two Stage Value Inference======================

FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT = """\
You are solving the FrozenLake puzzle. Forbid the hole and go to the target. You may move to the unintended direction due to the slippery ice. 
The thinking process must include an understanding of the game state, based on which the Assistant can later decide on the next move.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to only 1 actions.
"""

FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_WITHOUT_DYNAMIC_HINT = """\
You are solving the FrozenLake puzzle. Forbid the hole and go to the target.

The thinking process must include an understanding of the game state, based on which the Assistant can later decide on the next move.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to only 1 actions.
"""

FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_WITH_DYNAMIC_HINT = """\
You are solving the FrozenLake puzzle. Forbid the hole and go to the target.

## Environment Hint

1. The lake is slippery. There are 1/3 chance to slip into left or right direction. We cannot fall into hole.
2. We need to consider the most safe option that guarantee no falling into hole. If it is unavoidable, choose the most promising one.
3. You dont need to be worry about moving out-of-bound.
4. A example:

State:
_____
O_PO_
O____
___G_
O____

Reasoning: Since the lake is slippery, moving up can risk slipping to right and fall into hall. Moving down can also risk slipping to right and fall into hall.
So the most safe move is left as slipping to up or down are both fine. So the move is Left.

5. You will be given the probability of falling into hole for each move due to slippery dynamics. Take that into consideration when deciding the next move.

6. The thinking process must include an understanding of the game state, based on which the Assistant can later decide on the next move.
7. The meaning of each symbol in the state is: P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
8. Your available actions are:
Left, Down, Right, Up
9. You can make up to only 1 actions.
"""

FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_WITH_DYNAMIC_HINT2="""\
You are solving the FrozenLake puzzle. Forbid the hole and go to the target.

## Environment Hint
1. The meaning of each symbol in the state is: P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
2. Your available actions are: Left, Down, Right, Up
3. You can make up to only 1 actions.
4. The lake is slippery. There are 1/3 chance to slip into left or right direction. We cannot fall into hole.
5. At each state, you can compute the probability of falling into hole for each move due to slippery dynamics. For example, for state:
_____
O_PO_
O____
___G_
O____

The probability of falling into hole for each move is:
Left: 0. (slipping to up and down are both safe)
Down: 1/3 (slipping to right will fall into hole)
Right: 1/3 (when no slipping, moving right will fall into hole)
Up: 1/3 (slipping to right will fall into hole)

6. You dont need to be worry about moving out-of-bound.
7. Make sure to be safe first, then consider the goal.
"""


FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT = """\
Turn {turn_idx}:
State:
{state}
You have {turn_left} turns left. 
"""

STATE_VALUE_QUERY_PROMPT="""\
What is your understanding of the game state:

{state}
"""

STATE_ANSWER_DIRECT_PROMPT="""
Based on your understanding of the game state, what is the decided move?

Output your move in a form of <answer>[your move]</answer>
"""  # for direct inference


STATE_ANSWER_DIRECT_PROMPT_WITH_OPTION="""
Based on your understanding of the game state, what is the decided move?

Choose your move from ONE of the following options:
- <answer>Left</answer>
- <answer>Down</answer>
- <answer>Right</answer>
- <answer>Up</answer>
"""  # for direct inference with options

QWEN_TWO_STAGE_VALUE_PROMPT_SET = {
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT,
    "query_prompt": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "query_prompt_with_info": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}

QWEN_TWO_STAGE_VALUE_PROMPT_SET_WITHOUT_DYNAMIC_HINT={
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_WITHOUT_DYNAMIC_HINT,
    "query_prompt": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "query_prompt_with_info": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "knowledge_update_prompt": ADVANCED_KNOWLEDGE_UPDATE_PROMPT
}

QWEN_TWO_STAGE_VALUE_PROMPT_SET_WITH_DYNAMIC_HINT={
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_WITH_DYNAMIC_HINT,
    "query_prompt": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "query_prompt_with_info": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "knowledge_update_prompt": KNOWLEDGE_UPDATE_PROMPT
}

QWEN_TWO_STAGE_VALUE_PROMPT_SET_WITH_DYNAMIC_HINT2={
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_WITH_DYNAMIC_HINT2,
    "query_prompt": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "query_prompt_with_info": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "knowledge_update_prompt": KNOWLEDGE_UPDATE_PROMPT
}

FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_CLEANEST = """\
You are solving the FrozenLake puzzle. Forbid the hole and go to the target.
Try to recall [Environment Hint of FrozenLake] and think with it.
"""


FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_CLEANEST2 = """\
You are solving the FrozenLake puzzle. Forbid the hole and go to the target.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available actions are:
Left, Down, Right, Up
"""

STATE_VALUE_QUERY_PROMPT_WITH_HINT="""\
Try to recall [Environment Hint of FrozenLake]. What is your understanding of the game state:

{state}
"""

QWEN_TWO_STAGE_VALUE_PROMPT_SET_CLEANEST={
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_CLEANEST2,
    "query_prompt": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "query_prompt_with_info": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT, #STATE_VALUE_QUERY_PROMPT_WITH_HINT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "knowledge_update_prompt": KNOWLEDGE_UPDATE_PROMPT,
    "knowledge_update_prompt_TD": KNOWLEDGE_UPDATE_PROMPT_TD,
}


STATE_VALUE_QUERY_PROMPT_WITH_RECALL="""\
What is your understanding of the game state:

{state}

Also try to recall hints in a format of <FrozenLake Hint> [Hint] </FrozenLake Hint> among your reasoning process.
"""


QWEN_TWO_STAGE_VALUE_PROMPT_SET_CLEANEST_RECALL={
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_CLEANEST2,
    "query_prompt": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "query_prompt_with_info": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT_WITH_RECALL, #STATE_VALUE_QUERY_PROMPT_WITH_HINT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT_WITH_OPTION,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "knowledge_update_prompt": KNOWLEDGE_UPDATE_PROMPT
}

# =============== Two Stage LPM =====================


FROZENLAKE_TWO_STAGE_VALUE_LPM_SYSTEM_PROMPT = """\
You are solving the FrozenLake puzzle. Forbid the whole and go to the target. You may move to the unintended direction due to the slippery ice. 
You can plan a few steps ahead by interacting with a simulator, and then think about the reasoning process and at last provides the answer.

The thinking process must include an understanding of the game state, based on which the Assistant can later decide on the next move.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to only 1 actions.
"""

VALUE_CENTRIC_PARALLEL_ACT_QUERY_PROMPT="""
Current Game State:
{state}
"""
VALUE_CENTRIC_PARALLEL_PLAN_QUERY_PROMPT="""\
Current Simulation State:
{state}
"""
TWO_STAGE_VALUE_SUMMARIZE_PLAN_QUERY_PROMPT="""\
The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 
"""
PLAIN_PLAN_SUCCESS_PROMPT="""\
Results: You have successfully solve the game within this simulation!
"""
PLAIN_PLAN_FAIL_PROMPT="""\
The next state of current simulation:
{state}
Results: You lose the game within this simulation.
"""
TWO_STAGE_VALUE_SUMMARY_PROMPT="""\
[Summarizing stage]

You are an evaluation agent analyzing a set of simulation trajectories that originate from the same root state.
Your goal is to conceptually emulate value-iteration-like reasoning:

### Analysis Guidelines

1. The simulation history contains multiple independent trajectories. Each trajectory is a sequence of turns represented by tuples:
    (state, action, environment_reward, next_state, game_terminated)
2. Use the simulation rollouts as noisy signals of the action quality.
3. Examine the successor states reached by each action.
4. Combine the rollout evidence with your chess understanding to estimate the long-term value of each root action.
5. You are not required to compute numerical values—only conceptual judgments.
6. The simulations may not contain wins or terminal rewards, so you must infer potential future value from position quality.

### Simulation History

{history}
"""
SIMULATION_STATE_VALUE_QUERY_PROMPT="""\
Based on the simulation history, what is your updated understanding of the game state:

{state}
"""
PARALLEL_PLAN_STATE_MOVE_QUERY_PROMPT="""\
You are now in simulation, your current goal is to move in order to better understand potential future outcomes of the game, so you can either explore or exploit. 
You can generate at most {num_actions} candidate moves to be tested with the simualtor.
Do not generate duplicate moves.
Choose the move from available move list:
{available_move}

Based on your previous understanding of the game state and your current goal, generate moves in a form of:

<move>[Your First Candidate Move]</move>
<move>[Your Second Candidate Move]</move>
...
"""
STATE_ANSWER_QUERY_PROMPT="""\
The simulation ends, your current goal is to output the optimal move for the root state.

Based on previous simulation, your understanding of the game state and your current goal, what is the best move to make?

Output your move in a form of <answer>[Your Move]</answer>
"""


QWEN_TWO_STAGE_VALUE_PARALLEL_PLAN_PROMPT_SET={
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_LPM_SYSTEM_PROMPT,
    "act_query_prompt": VALUE_CENTRIC_PARALLEL_ACT_QUERY_PROMPT,
    "plan_query_prompt": VALUE_CENTRIC_PARALLEL_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_info": VALUE_CENTRIC_PARALLEL_PLAN_QUERY_PROMPT,
    "plan_action_query_prompt": TWO_STAGE_VALUE_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAIN_PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAIN_PLAN_FAIL_PROMPT,
    "independent_summary_prompt": TWO_STAGE_VALUE_SUMMARY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "simulation_state_value_query_prompt": SIMULATION_STATE_VALUE_QUERY_PROMPT,
    "state_move_query_prompt": PARALLEL_PLAN_STATE_MOVE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_QUERY_PROMPT,
}





# Reflexion with long-cot understanding reaosning

FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_REFLEXION="""
You are solving the FrozenLake puzzle. Forbid the hole and go to the goal.
Example answer format: <think>To forbid the hole and go to the goal, I should go left then go up.</think><answer>Left</answer>

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to only 1 actions.
"""


QWEN_TWO_STAGE_VALUE_PROMPT_SET_CLEANEST_FOR_REFLEXION={
    "system_prompt": FROZENLAKE_TWO_STAGE_VALUE_SYSTEM_PROMPT_REFLEXION,
    "query_prompt": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "query_prompt_with_info": FROZENLAKE_TWO_STAGE_VALUE_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT, #STATE_VALUE_QUERY_PROMPT_WITH_HINT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "reflexion_prompt": REFLEXION_PROMPT
}