import numpy as np
import re
import random



def chess_extract_best_move(response, available_actions):
    """

    Args:
        response:
        available_actions: actually a dict

    Returns:

    """
    enable_think = False
    special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
    max_actions=1
    # action_lookup = {"left": 1, "down": 2, "right": 3, "up": 4} #{1:"Left", 2:"Down", 3:"Right", 4:"Up"}
    action_sep = ","

    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return random.choice(available_actions), 0.

    else:
        if enable_think:
            think_content, action_content = match.group(1), match.group(2)
        else:
            think_content, action_content = "", match.group(1)

        for special_token in special_token_list:
            action_content = action_content.replace(special_token, "").strip()
            think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]

        if len(actions) >= max_actions:
            ret_action = (actions[0])

            if ret_action in available_actions:
                return ret_action, 1.0      # output action string
            else:
                return random.choice(available_actions), 0.

        else:
            return random.choice(available_actions), 0.


def chess_extract_best_plan_move(response, available_actions):
    """

    Args:
        response:
        available_actions: actually a dict

    Returns:

    """
    enable_think = False
    special_token_list = ["<think>", "</think>", "<move>", "</move>", "<|im_start|>", "<|im_end|>", "+", "#"]   # san move leak symbol
    max_actions=1
    action_sep = ","
    lower_letter_available_actions = [i.lower() for i in available_actions]

    pattern = r'<think>(.*?)</think>\s*<move>(.*?)</move>' if enable_think else r'<move>(.*?)</move>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
        # return random.choice(available_actions), 0.
        # llm_response, actions = response, []
        raise ValueError(f"From response {response} \naction is not deteted in available action list {lower_letter_available_actions}")

        # ret_action = available_actions[
        #     int(np.random.choice(len(available_actions), 1)[0])]
        # return ret_action, 0.
    else:
        if enable_think:
            think_content, action_content = match.group(1), match.group(2)
        else:
            think_content, action_content = "", match.group(1)

        for special_token in special_token_list:
            action_content = action_content.replace(special_token, "").strip()
            think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        # if len(actions) > max_actions:
        #     actions = actions[:max_actions]  # Only the first MAX_ACTIONS actions are kept in the rollout.
        #     action_content = (" " + self.action_sep + " ").join(actions)
        if len(actions) >= max_actions:
            ret_action = (actions[0])
            # retrive action index
            if ret_action.lower() in lower_letter_available_actions:
                # return ret_action, 1.0, ""      # output action string?
                position_idx = lower_letter_available_actions.index(ret_action.lower())
                return available_actions[position_idx], 1.0, ""

            else:
                # raise NotImplementedError(f"action {ret_action} not in available list {available_actions}")
                return random.choice(available_actions), 0., "[Move Error] The output move is not in the available list."

        else:
            # raise NotImplementedError(f"action {ret_action} not in available list {available_actions}")
            return random.choice(available_actions), 0., "[Move Error] The output move is not valid"


def chess_mdp_extract_best_answer(response_text, action_pattern, available_actions):
    if response_text is None:
        return random.choice(available_actions), 0.
    
    match = action_pattern.search(response_text)
    if not match:
        return random.choice(available_actions), 0.
    else:
        action_content = match.group(1).strip()
    
    lower_letter_available_actions = [i.lower() for i in available_actions]
    special_token_list = ["<think>", "</think>", "<move>", "</move>", "<|im_start|>", "<|im_end|>", "+", "#"]
    for special_token in special_token_list:
        action_content = action_content.replace(special_token, "").strip()
    
    if action_content.lower() in lower_letter_available_actions:
        position_idx = lower_letter_available_actions.index(action_content.lower())
        return available_actions[position_idx], 1.0
    else:
        return random.choice(available_actions), 0.

    