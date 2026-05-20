SCI_SYSTEM_PROMPT="""\
You are an agent, your job is to do some scientific experiment in a virtual text-based environment.

## Response Format:
You MUST use this exact format for every response. All tags are REQUIRED in sequential order:

<think>your analytical reasoning and thought process</think>
<answer>exactly one specific action command</answer>

## Notes:
At each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the <think> </think> tag and wrap your action with the <action> </action> tag.
You should ALWAYS take one action each step.
DO NOT try to interact with the user at anytime. Finish the task by yourself.

## Available Commands:
Below are the available commands you can use:
    open OBJ: open a container
    close OBJ: close a container
    activate OBJ: activate a device
    deactivate OBJ: deactivate a device
    connect OBJ to OBJ: connect electrical components
    disconnect OBJ: disconnect electrical components
    use OBJ [on OBJ]: use a device/item
    look around: describe the current room
    examine OBJ: examine an object in detail
    look at OBJ: describe a container's contents
    read OBJ: read a note or book
    move OBJ to OBJ: move an object to a container
    pick up OBJ: move an object to the inventory
    pour OBJ into OBJ: pour a liquid into a container
    mix OBJ: chemically mix a container
    teleport to LOC: teleport to a specific room
    focus on OBJ: signal intent on a task object
    wait: take no action for 10 steps
    wait1: take no action for a step

## Action Format Examples:
Your output should be like this:
<think>Now I will check the bedroom to find the thermometer...</think><answer>teleport to bedroom</answer>

<think>I need to examine the substance to understand its properties...</think><answer>examine substance</answer>

<think>To boil the water, I should activate the heating element...</think><answer>activate heating element</answer>
"""

SCI_QUERY_PROMPT="""\
{task_description}.

You are in the following state:
{state}
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



RL_PROMPT_SET={
    "system_prompt": SCI_SYSTEM_PROMPT,
    "query_prompt": SCI_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}



SYSTEM_PROMPT_V2="""\
You are an agent, your job is to do some scientific experiment in a virtual text-based environment. 

## Response Format: You MUST use this exact format for every response. All tags are REQUIRED in sequential order: 
<think>your analytical reasoning and thought process</think> 
<answer>exactly one specific action command</answer>  

## Notes: At each step, you should first think then perform action to fulfill the instruction. 
You should ALWAYS take one action each step. 
DO NOT try to interact with the user at anytime. Finish the task by yourself.  

## Available Commands: 
[Navigation] look, look around, look at OBJ, go to LOC, teleport to LOC 
[Interaction] open OBJ, close OBJ, pick up OBJ, put OBJ in CONTAINER, pour OBJ into CONTAINER 
[Task] focus on OBJ, wait, wait1
"""

SCI_QUERY_PROMPT_V2="""\
## Goal
{task_description}.

## Current State
{state}
"""

RL_PROMPT_SET_V2={
    "system_prompt": SYSTEM_PROMPT_V2,
    "query_prompt": SCI_QUERY_PROMPT_V2,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}

# ===== One Stage Parallel Planning Prompt Set

PARALLEL_PLAN_SYSTEM_PROMPT="""\
You are an agent, your job is to do some scientific experiment in a virtual text-based environment. 
At each state, you are able to simulate a few steps first and summarize the first move to take.

## Response Format: You MUST use this exact format for every response. All tags are REQUIRED in sequential order: 
<think>your analytical reasoning and thought process</think> 
<move>exactly one specific action command</move>  

## Notes: At each step, you should first think then perform action to fulfill the instruction. 
You should ALWAYS take one action each step. 
DO NOT try to interact with the user at anytime. Finish the task by yourself.  

## Available Commands: 
[Navigation] look, look around, look at OBJ, go to LOC, teleport to LOC 
[Interaction] open OBJ, close OBJ, pick up OBJ, put OBJ in CONTAINER, pour OBJ into CONTAINER 
[Task] focus on OBJ, wait, wait1
"""

PARALLEL_ACT_QUERY_PROMPT="""\
## Current Game State:
{state}

## Goal:
{task_description}
"""

PARALLEL_PLAN_QUERY_PROMPT="""\
Current Simulation State:
{state}


You can now plan for maximum {max_step} number of steps ahead. You can generate {num_actions} independent parallel actions for each state.
You can generate at most {num_actions} candidate moves to be tested with the simulator.
Do not generate duplicate moves.

Based on your understanding of the game state and your current goal, generate moves in a form of:
<think>thought</think><move>Your First Candidate action command</move>
<think>thought</think><move>Your Second Candidate action command</move>
...
"""

PLAIN_PLAN_SUCCESS_PROMPT="""\
Results: You have successfully solve the game within this simulation!
"""

PLAIN_PLAN_FAIL_PROMPT="""\
The next state of current simulation:
{state}
Results: You lose the game within this simulation.
"""

PARALLEL_PLAN_ACTION_QUERY_PROMPT="""\
The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 
"""


INDEPENDENT_SUMMARY_PROMPT="""\
[Summarizing stage]

The simulation ends. Lets pause, check the simulation history, and summarize from the simulation history to get the best FIRST move at root state.

## Goal
{task_description}

## Simulation History
{history}

## Root State
{state}
"""



PARALLEL_PLAN_SET={
    "system_prompt": PARALLEL_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": PARALLEL_ACT_QUERY_PROMPT,
    "plan_query_prompt": PARALLEL_PLAN_QUERY_PROMPT,
    "plan_action_query_prompt": PARALLEL_PLAN_ACTION_QUERY_PROMPT,
    "plan_success_prompt": PLAIN_PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAIN_PLAN_FAIL_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    # 
    # "independent_summary_instruction": PARALLEL_PLAN_SYSTEM_PROMPT,
    "independent_summary_prompt": INDEPENDENT_SUMMARY_PROMPT,

}


# ===== Planning Prompt Set =====



SCI_PLAN_SYSTEM_PROMPT="""\
You are an agent, your job is to do some scientific experiment in a virtual text-based environment.

## Response Format:
You MUST use this exact format for every response. All tags are REQUIRED in sequential order:

<think>your analytical reasoning and thought process</think>
<move>exactly one specific action command</move>


## Notes:
At each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the <think> </think> tag and wrap your action with the <action> </action> tag.
You should ALWAYS take one action each step.
DO NOT try to interact with the user at anytime. Finish the task by yourself.

## Available Commands:
Below are the available commands you can use:
    open OBJ: open a container
    close OBJ: close a container
    activate OBJ: activate a device
    deactivate OBJ: deactivate a device
    connect OBJ to OBJ: connect electrical components
    disconnect OBJ: disconnect electrical components
    use OBJ [on OBJ]: use a device/item
    look around: describe the current room
    examine OBJ: examine an object in detail
    look at OBJ: describe a container's contents
    read OBJ: read a note or book
    move OBJ to OBJ: move an object to a container
    pick up OBJ: move an object to the inventory
    pour OBJ into OBJ: pour a liquid into a container
    mix OBJ: chemically mix a container
    teleport to LOC: teleport to a specific room
    focus on OBJ: signal intent on a task object
    wait: take no action for 10 steps
    wait1: take no action for a step

## Action Format Examples:
Your output should be like this:

<think>Now I will check the bedroom to find the thermometer...</think><move>teleport to bedroom</move>

<think>I need to examine the substance to understand its properties...</think><move>examine substance</move>

<think>To boil the water, I should activate the heating element...</think><move>activate heating element</move>
"""

SCI_ACT_QUERY_PROMPT="""\
[Game Step {turn_idx}]

Game State:
{state}
"""

SCI_PLAN_QUERY_PROMPT="""\
[Simulation turn {turn_idx}, Simulation step {step_idx}]

Game State:
{state}
"""

SCI_PLAN_ACTION_QUERY_PROMPT="""\
[Summarizing stage]

The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 
Refer to previous simulation history for possible outcome of the the possible move.

# Goal
{task_description}

# Root State
{state}
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
This simulation turn ends due to invalid action. We will restart a new simulation turn.
"""

PLAN_FAIL_PROMPT_WITH_FEEDBACK="""\
Feedback for your previous move:
{previous_feedback}

The final board position of this simulation:
{state}
{extra_info}
The simulation ends {fail_reason}.
"""

INDEPENDENT_SUMMARY_PROMPT="""\
[Summarizing stage]

The simulation ends. Lets pause, check the simulation history, and summarize from the simulation history to get the best FIRST move at root state.

# Goal
{task_description}

# Simulation History
{history}

# Root State
{state}
"""


PLAN_PROMPT_SET={
    "system_prompt": SCI_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": SCI_ACT_QUERY_PROMPT,
    "plan_query_prompt": SCI_PLAN_QUERY_PROMPT,
    "plan_action_query_prompt": SCI_PLAN_ACTION_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,  # extra
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,  # extra
    "independent_summary_instruction": SCI_PLAN_SYSTEM_PROMPT,
    "independent_summary_prompt": INDEPENDENT_SUMMARY_PROMPT,
}