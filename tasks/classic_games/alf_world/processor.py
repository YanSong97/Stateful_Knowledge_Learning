import numpy as np
import re



def extract_best_move(response, available_actions):
    """
    Use in NLRL
    """
    try:
        if isinstance(response, str):
            result = response.split("""\"best_move\": """)[1][0]
        else:  # OpenAI ChatCompletion response
            response_text = response.choices[0].message.content.strip()
            result = response_text.split("""\"best_move\": """)[1].split("}")[0].strip()

        result = int(result)

        assert (
            result in available_actions
        ) #f"Error: {result} not in {available_actions}, response: {response}"
        valid_action = 1.
    except Exception as e:
        result = available_actions[
            int(np.random.choice(len(available_actions), 1)[0])
        ]
        # print(
        #     f"Error: random action selected {result} from {available_actions}, response: {response}\
        #         \nexception {e}\n"
        # )
        valid_action = 0.
    return result, valid_action


def extract_planning_move(response, available_actions):
    action_lookup = {"left": 1, "down": 2, "right": 3, "up": 4}
    enable_think = False
    special_token_list = ["<think>", "</think>", "<move>", "</move>", "<|im_start|>", "<|im_end|>"]
    max_actions=1
    action_sep = ","

    pattern = r'<think>(.*?)</think>\s*<move>(.*?)</move>' if enable_think else r'<move>(.*?)</move>'
    match = re.search(pattern, response, re.DOTALL)
    assert match, print(f"response should have move tag in it, {response}")

    if enable_think:
        think_content, action_content = match.group(1), match.group(2)
    else:
        think_content, action_content = "", match.group(1)

    for special_token in special_token_list:
        action_content = action_content.replace(special_token, "").strip()
        think_content = think_content.replace(special_token, "").strip()

    actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
    if len(actions) >= max_actions:
        ret_action = (actions[0]).lower()
        # retrive action index
        if ret_action in action_lookup:
            return action_lookup[ret_action], 1.0
        else:
            ret_action = available_actions[
                int(np.random.choice(len(available_actions), 1)[0])]
            return ret_action, 0.
    else:
        ret_action = available_actions[
            int(np.random.choice(len(available_actions), 1)[0])]
        return ret_action, 0.




def ragen_extract_best_move(response, action_pattern, available_actions):
    enable_think = True
    special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
    max_actions=1
    action_lookup = {"left": 1, "down": 2, "right": 3, "up": 4} #{1:"Left", 2:"Down", 3:"Right", 4:"Up"}
    action_sep = ","

    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
        # llm_response, actions = response, []
        ret_action = available_actions[
            int(np.random.choice(len(available_actions), 1)[0])]
        return ret_action, 0.
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
            ret_action = (actions[0]).lower()
            # retrive action index
            if ret_action in action_lookup:
                return action_lookup[ret_action], 1.0
            else:
                ret_action = available_actions[
                    int(np.random.choice(len(available_actions), 1)[0])]
                return ret_action, 0.
        else:
            ret_action = available_actions[
                int(np.random.choice(len(available_actions), 1)[0])]
            return ret_action, 0.


def ragen_nothink_extract_best_move(response, action_pattern, available_actions):
    enable_think = False
    special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
    max_actions=1
    action_lookup = {"left": 1, "down": 2, "right": 3, "up": 4} #{1:"Left", 2:"Down", 3:"Right", 4:"Up"}
    action_sep = ","

    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
        # llm_response, actions = response, []
        ret_action = available_actions[
            int(np.random.choice(len(available_actions), 1)[0])]
        return ret_action, 0.
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
            ret_action = (actions[0]).lower()
            # retrive action index
            if ret_action in action_lookup:
                return action_lookup[ret_action], 1.0
            else:
                ret_action = available_actions[
                    int(np.random.choice(len(available_actions), 1)[0])]
                return ret_action, 0.
        else:
            ret_action = available_actions[
                int(np.random.choice(len(available_actions), 1)[0])]
            return ret_action, 0.