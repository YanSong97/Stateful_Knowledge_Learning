"""SYSTEM:
You are an intelligent agent that interacts with a text-based simulated environment.
Your job is to output a sequence of actions that accomplish the given task.

Rules:
- Only output valid actions from the environment’s action set.
- Use simple structured output (one action per line or as a JSON list).
- Think step by step before choosing each action.

Example valid format:
1. go to fridge 1
2. take apple 1
...
Do not add extra explanations beyond the required actions.

USER:
You are in [CURRENT STATE DESCRIPTION].
Your goal: [TASK DESCRIPTION]."""


# ====================== AlfWorld Direct Inference======================
ALFWORLD_POLICY_SIMPLE_PROMPT_RAGEN = """\
You are an intelligent agent that interacts with a text-based simulated environment.
Your job is to output a sequence of actions that accomplish the given task.
Rules:
- Only output valid actions from the environment’s action set.
- Use simple structured output.
- Think step by step before choosing each action.
Example valid format for an action:
1. <think> [Your thoughts] </think> <answer> go to fridge 1 </answer>
2. <think> [Your thoughts] </think> <answer> take apple 1 </answer>
...
"""

ALFWORLD_STATE_USER_PROMPT_RAGEN = """\
Turn {turn_idx}:
State:
{state}
Available actions:
{available_move}
You have {turn_left} turns left. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.
"""

SUCCESS_PROMPT = """\
The final state:
{state}

The game ends. You have successfully solve the game! Congratulation!
"""

FAIL_PROMPT = """\
The final state:
{state}

The game ends. You did not solve the game {fail_reason}.
"""



RL_PROMPT_SET = {
    "system_prompt": ALFWORLD_POLICY_SIMPLE_PROMPT_RAGEN,
    "query_prompt": ALFWORLD_STATE_USER_PROMPT_RAGEN,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}



# ====================== https://arxiv.org/pdf/2505.10978 ======================

ALFWORLD_SYSTEM_PROMPT="""\
You are an expert agent operating in the ALFRED embodied Environment.
Your job is to output a sequence of actions that accomplish the given task.
Rules:
- Only output valid actions from the environment’s action set.
- Use simple structured output.
- Think step by step before choosing each action.
Example valid format for an action:
1. <think> [Your thoughts] </think> <answer> go to fridge 1 </answer>
2. <think> [Your thoughts] </think> <answer> take apple 1 </answer>
...
"""

ALFWORLD_USER_QUERY="""\
Your task is to: 
{task_description}.

You are in the following state:
{state}

Available actions:
{available_move}

You have {turn_left} turns left. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.
"""


RL_PROMPT_SET_V2 = {
    "system_prompt": ALFWORLD_SYSTEM_PROMPT,
    "query_prompt": ALFWORLD_USER_QUERY,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}

# ==== Detailed Inference ====

ALF_DETAILED_SYSTEM_PROMPT="""\
You are an agent interacting with a virtual text-based environment.

## Response Format:
You MUST use this exact format for every response. Both tags are REQUIRED in sequential order:

<think>your analytical reasoning and thought process</think>
<answer>exactly one specific action command</answer>

## Action Commands:
Your <answer> must be one of the following, strictly following the command (argument) format.

### Navigation & Observation:
- look: Look around your current location to get more details.
- inventory: Check the object you are currently holding (you can only hold one).
- go to (receptacle): Move to a receptacle (e.g., table, fridge, sink).

### Interacting with Receptacles:
- open (receptacle): Open a receptacle.
- close (receptacle): Close a receptacle.

### Interacting with Objects:
- take (object) from (receptacle): Pick up an object from a receptacle.
- move (object) to (receptacle): Place the object you are holding into or onto a receptacle.
- examine (object): Examine an object closely to learn its properties.

### Changing Object States:
- heat (object) with (receptacle): Heat an object with a device (e.g., microwave).
- cool (object) with (receptacle): Cool an object with a device (e.g., fridge).
- clean (object) with (receptacle): Clean an object with a device (e.g., sink).
- slice (object) with (object): Slice an object using a sharp object (e.g., knife).

For example your output should be like this:
<think>your reasoning process here</think>
<answer>look</answer>

<think>your reasoning process here</think>
<answer>go to sofa 1</answer>

## Critical Rules & Constraints
- Single Item Inventory: You can only hold one object at a time. You must put down the current object before taking a new one.
- Examine Before Acting: Before performing an action on an object (like take, heat, or clean), it is best to examine it first to confirm its properties.
- Use Exact Names: The (object) and (receptacle) arguments in your command MUST exactly match the names seen in your Observation, including any numbers (e.g., apple 1, desk 2).
- Systematic Thinking: Break down complex tasks into smaller, manageable sub-goals. Clearly outline your plan in the <think> block.
- Step Limit: You must complete the task within 25 steps.
"""

ALF_DETAILED_USER_QUERY="""\
Your task is to: 
{task_description}.

You are in the following state:
{state}
"""


RL_PROMPT_SET_V3={
    "system_prompt": ALF_DETAILED_SYSTEM_PROMPT,
    "query_prompt": ALF_DETAILED_USER_QUERY,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}


# ====== Policy-centric Sequential Infer ====


QWEN_PLAN_SIMPLEST_PROMPT="""\
You are an expert agent operating in the ALFRED embodied Environment.
Your job is to output a sequence of actions that accomplish the given task.
Rules:
- Only output valid actions from the environment’s action set.
- Use simple structured output.
- Think step by step before choosing each action.

During Simulation stage, you need to output format: <think> [Your thoughts] </think> <move> [your answer] </move> with no extra text. Strictly follow this format.
During Summarizing stage, you need to output format: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.
"""

ACT_QUERY_PROMPT_CLEAN="""\
[Game Step {turn_idx}]

Your task is to: 
{task_description}.

Game State:

{state}
"""

PLAN_QUERY_PROMPT_CLEAN="""\
[Simulation turn {turn_idx}, Simulation step {step_idx}, Planning stage]

Game State:

{state}

Available actions:
{available_move}

Simulation budgets left:
{turn_left} turn budget left for resetting the simulation. {max_step} steps for planning ahead.

Always output: <think> [Your thoughts] </think> <move> [your answer] </move>
"""

SUMMARIZE_QUERY_PROMPT_CLEAN="""\
[Summarizing stage]

The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 
Refer to previous simulation history for possible outcome of the the possible move.

## State

{state}

## Goal
Your task is to: 
{task_description}.

## Available Actions
{available_move}

## Output format
<think> [Your thoughts] </think> <answer> [your answer] </answer>
"""

PLAN_SUCCESS_PROMPT="""\
The final board position of this simulation turn:
{state}

This simulation turn ends. You have successfully solve the game within this simulation! Congratulation! We will reset to Simulation Step 1.
"""

PLAN_FAIL_PROMPT="""\
The final board position of this simulation turn:
{state}
{extra_info}
This simulation turn ends {fail_reason}. We will start a new simulation turn.
"""

PLAN_FAIL_PROMPT_WITH_FEEDBACK="""\
Feedback for your previous move:
{previous_feedback}

The final board position of this simulation:
{state}
{extra_info}
The simulation ends {fail_reason}.
"""

INDEPENDENT_SUMMARY_INSTRUCTION="""\
You are an expert agent operating in the ALFRED embodied Environment.
Your job now is to analyze the simulation history and summarize the best move at root (initial) state.

- Make sure your move is in the available actions list!
- Make sure your move is for the root state only! DO not confuse the root state with the state in the simulation history!
"""

INDEPENDENT_SUMMARY_PROMPT="""\
[Summarizing stage]

The simulation ends. Lets pause, check the simulation history, and summarize from the simulation history to get the best move at root state.

## Goal
Your task is to: 
{task_description}.

## Simulation History
{history}

## Root State
{state}

## Output format
<think> Your summary </think> <answer>Your Move at root state</answer>

## Available Actions
{available_move}
"""

## History before root state
#{hisotry_before_root_state}

QWEN_SIMPLEST_PLAN_PROMPT_SET={
    "system_prompt": QWEN_PLAN_SIMPLEST_PROMPT,
    "act_query_prompt": ACT_QUERY_PROMPT_CLEAN,
    "act_query_prompt_with_info": ACT_QUERY_PROMPT_CLEAN,
    "plan_query_prompt": PLAN_QUERY_PROMPT_CLEAN,
    "plan_query_prompt_with_info": PLAN_QUERY_PROMPT_CLEAN,
    "plan_query_prompt_with_feedback": PLAN_QUERY_PROMPT_CLEAN,
    "plan_action_query_prompt":SUMMARIZE_QUERY_PROMPT_CLEAN,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,  # extra
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,  # extra
    "independent_summary_instruction": INDEPENDENT_SUMMARY_INSTRUCTION,
    "independent_summary_prompt": INDEPENDENT_SUMMARY_PROMPT,
}


# ====== Policy-centric Parallel Infer ====

QWEN_POLICY_PARALLEL_PROMPT="""\
You are an expert agent operating in the ALFRED embodied Environment.
Your job is to output a sequence of actions that accomplish the given task.
Rules:
- Only output valid actions from the environment’s action set.
- Use simple structured output.
- Think step by step before choosing each action.

During Simulation stage, you need to output format: <think> [Your thoughts] </think> <move> [your answer] </move> with no extra text. Strictly follow this format.
During Summarizing stage, you need to output format: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.
"""

QWEN_POLICY_PARALLEL_PROMPT_SET={
    "system_prompt": QWEN_POLICY_PARALLEL_PROMPT,
}




# ====== Value infer ====

TWO_STAGE_VALUE_SYSTEM_PROMPT="""\
You are an expert agent operating in the ALFRED embodied Environment.
Your job is to output a sequence of actions that accomplish the given task.
Rules:
- Only output valid actions from the environment’s action set.
- Use simple structured output.
- Think step by step before choosing each action.

You will be first ask about your understanding of the game state, and then ask about the best move to make.
"""

TWO_STAGE_VALUE_ACT_QUERY_PROMPT="""\
Your task is to: 
{task_description}.

You are in the following state:
{state}

Available actions:
{available_move}
"""

STATE_VALUE_QUERY_PROMPT="""\
What is your understanding of the game state:

{state}
"""

STATE_ANSWER_DIRECT_PROMPT="""
Based on your understanding of the game state, what is the decided move?

Output your move in a form of <answer>[Your Move]</answer>
"""  # for direct inference


QWEN_TWO_STAGE_VALUE_DIRECT_PROMPT_SET = {
    "system_prompt": TWO_STAGE_VALUE_SYSTEM_PROMPT,
    "query_prompt": TWO_STAGE_VALUE_ACT_QUERY_PROMPT,
    "query_prompt_with_info": TWO_STAGE_VALUE_ACT_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}


# ====== Value Parallel Inference ====

QWEN_VALUE_PARALLEL_SYSTEM_PROMPT="""\
You are an expert agent operating in the ALFRED embodied Environment.
Your job is to output a sequence of actions that accomplish the given task.
Rules:
- Only output valid actions from the environment’s action set.
- Use simple structured output.
- Think step by step before choosing each action.

You will be first ask about your understanding of the game state, and then ask about the best move to make.
"""

QWEN_VALUE_PARALLEL_ACT_QUERY_PROMPT="""\
Current State:
{state}
"""

QWEN_VALUE_PARALLEL_PLAN_QUERY_PROMPT="""\
Current Simulation State:
{state}
"""

QWEN_VALUE_PARALLEL_PLAN_QUERY_PROMPT_WITH_INFO="""\
The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 
"""

QWEN_VALUE_PARALLEL_SUMMARY_HISTORY_PROMPT="""\
The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 

# Simulation History

{history}
"""

STATE_VALUE_QUERY_PROMPT="""\
What is your understanding of the game state:

{state}
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



QWEN_TWO_STAGE_VALUE_PARALLEL_PROMPT_SET={
    "system_prompt": QWEN_VALUE_PARALLEL_SYSTEM_PROMPT,
    "act_query_prompt": QWEN_VALUE_PARALLEL_ACT_QUERY_PROMPT,
    "plan_query_prompt": QWEN_VALUE_PARALLEL_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_info": QWEN_VALUE_PARALLEL_PLAN_QUERY_PROMPT,
    "plan_action_query_prompt": QWEN_VALUE_PARALLEL_PLAN_QUERY_PROMPT_WITH_INFO,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "independent_summary_prompt": QWEN_VALUE_PARALLEL_SUMMARY_HISTORY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "simulation_state_value_query_prompt": SIMULATION_STATE_VALUE_QUERY_PROMPT,
    "state_move_query_prompt": PARALLEL_PLAN_STATE_MOVE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_QUERY_PROMPT
}
