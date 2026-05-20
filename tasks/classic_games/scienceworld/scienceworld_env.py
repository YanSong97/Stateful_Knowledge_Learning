
import os
import json
from pathlib import Path
import random
import asyncio
from scienceworld import ScienceWorldEnv

class ScienceWorldEnv_Wrapper:
    def __init__(self, env_config, **kwargs):
        self.config = env_config

        train_data_path = Path(__file__).parent / "sciworld_data" / f"train.jsonl"
        train_data = []
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    item = json.loads(line)
                    train_data.append(item)
        self.train_data = train_data

        test_data_path = Path(__file__).parent / "sciworld_data" / f"test.jsonl"
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    item = json.loads(line)
                    test_data.append(item)
        self.test_data = test_data

        self.max_steps = env_config.max_steps   # 30
        self.current_step = 0
        
        self.num_train = 2294
        self.num_test = 1308
        self.jar_path = self._get_config_value("jar_path") or os.environ.get("SCIENCEWORLD_JAR_PATH")

    def _get_config_value(self, key, default=None):
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return getattr(self.config, key, default)
    
    @property
    def num_game(self):
        return 100
        # return self.num_train + self.num_test


    def reset(self, seed=None, specified_game_idx=None):
        self.specific_game_idx = specified_game_idx

        if specified_game_idx is None:
            specified_game_idx = random.randint(0, self.num_game - 1)

        if specified_game_idx < self.num_test:
            data_instance = self.test_data[specified_game_idx]
        else:
            data_instance = self.train_data[specified_game_idx - self.num_test]

        task_config = json.loads(data_instance["task_desc"])
        task_name = task_config["task_name"]
        variation = task_config.get("var_num", 0)
        jar_path = self.jar_path or task_config.get("jar_path", "")
        if not jar_path:
            raise ValueError(
                "ScienceWorld jar path is required. Set env_config.jar_path or SCIENCEWORLD_JAR_PATH."
            )

        # if hasattr(self, 'env') and self.env is not None:
        #     self.env.close()
        #     try:
        #         # Try to close the Java gateway connection
        #         if hasattr(self.env, 'shutdown'):
        #             self.env.shutdown()
        #         elif hasattr(self.env, 'close'):
        #             self.env.close()
        #     except Exception as e:
        #         # Log but don't fail if cleanup fails
        #         import logging
        #         logging.warning(f"Failed to close previous ScienceWorldEnv: {e}")

        
        self.env = ScienceWorldEnv("", jar_path, envStepLimit=30)
        simplificationStr = "easy"
        self.env.load(task_name, variation, simplificationStr, generateGoldPath=True)

        observation, info = self.env.reset()
        self.last_obs = observation
        self.last_info = info
        self.task_goal = self.env.get_task_description()
        self.full_action_history = []
        self.current_step = 0
        self.game_status = None
        self.reward_record = 0
        self.final_reward = 0

        return self.observe()
    
    def get_key_stats(self):
        return {
            "game_idx": self.specific_game_idx,
            "last_obs": self.last_obs,
            "last_info": self.last_info,
            "task_description": self.task_goal,
            "full_action_history": [i for i in self.full_action_history],
        }

    def reset_state(self, state_info):
        game_idx = state_info["game_idx"]
        self.reset(specified_game_idx=game_idx)
        
        for a in state_info["full_action_history"]:
            _ = self.step(a)
        return self.observe()

    def observe(self):
        return self.last_obs
    
    @property
    def legal_moves_list(self):
        """
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
        """
        move_hint = [
            "open <OBJ>: open a container",
            "close <OBJ>: close a container",
            "activate <OBJ>: activate a device",
            "deactivate <OBJ>: deactivate a device",
            "connect <OBJ> to <OBJ>: connect electrical components",
            "disconnect <OBJ>: disconnect electrical components",
            "use <OBJ> [on <OBJ>]: use a device/item",
            "look around: describe the current room",
            "examine <OBJ>: examine an object in detail",
            "look at <OBJ>: describe a container's contents",
            "read <OBJ>: read a note or book",
            "move <OBJ> to <OBJ>: move an object to a container",
            "pick up <OBJ>: move an object to the inventory",
            "pour <OBJ> into <OBJ>: pour a liquid into a container",
            "mix <OBJ>: chemically mix a container",
            "teleport to <LOC>: teleport to a specific room",
            "focus on <OBJ>: signal intent on a task object",
            "wait: take no action for 10 steps",
            "wait1: take no action for a step",
        ]
        # return move_hint
        return []

    @property
    def legal_moves(self):
        return self.legal_moves_list
    
    @property
    def legal_moves_string(self):
        return "[" + ";\n ".join(self.legal_moves_list) + "]"

    @property
    def legal_moves_string(self):
        return ", ".join(self.legal_moves_list)

    def step(self, action):
        self.current_step += 1
        self.full_action_history.append(action)
        observation, reward, done, info = self.env.step(action)
        reward /= 100
        self.last_obs = observation
        self.last_info = info
        if reward > self.reward_record:
            self.reward_record = reward

        if self.current_step >= self.max_steps:
            done = True

        if done:
            if reward > 0.:
                self.game_status = "win"
            else:
                self.game_status = "lose"

        return self.observe(), reward, done, info
    
    def check_success(self):
        return self.game_status == "win"
    
    def check_lose(self):
        return self.game_status == "lose"

    def check_results(self):
        return self.game_status
    
    def close(self):
        self.env.close()


class AsyncScienceWorldEnv_Wrapper(ScienceWorldEnv_Wrapper):
    async def step(self, action):
        self.current_step += 1
        self.full_action_history.append(action)
        observation, reward, done, info = self.env.step(action)
        reward /= 100
        self.last_obs = observation
        self.last_info = info
        if reward > self.reward_record:
            self.reward_record = reward

        if self.current_step >= self.max_steps:
            done = True

        if done:
            if reward > 0.:
                self.game_status = "win"
            else:
                self.game_status = "lose"

        return self.observe(), reward, done, info
    
    async def reset_state(self, state_info):
        game_idx = state_info["game_idx"]
        self.reset(specified_game_idx=game_idx)
        
        for a in state_info["full_action_history"]:
            _ = await self.step(a)
        return self.observe()



if __name__ == "__main__":

    # export _JAVA_OPTIONS="-Xmx500g -Xms1g"
    
    class EnvConfig:
        max_steps = 30
    
    cfg = EnvConfig
    env = ScienceWorldEnv_Wrapper(cfg)
    init_obs = env.reset(specified_game_idx=1)

    print(f"init_obs = {init_obs}")

    ret = env.step("focus on glass cup")
    obs, reward, done, info = ret

    print(f"ret = {ret}")
    print(f"obs = {obs}")
    print(f"reward = {reward}")
    print(f"done = {done}")
    # print(f"info = {info}")

    print(f"env.game_status = {env.game_status}")
    print(f"env.final_reward = {env.final_reward}")

    # print(f"env shutdown = {env.env.shutdown()}")
    print(f"env close = {env.env.close()}")


    # # Get the directory where this script is located
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(script_dir, "sciworld_data")
    # data_type = 'train'  # 'test'

    # data_file = os.path.join(data_path, f"{data_type}.jsonl")

    
    # data = []
    # with open(data_file, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         if line:  # Skip empty lines
    #             item = json.loads(line)
    #             data.append(item)
    
    # # Process each item in the loaded data
    # env = None
    # current_jar_path = None

    # # data[0] is already a dict, extract the task_desc field (which is a JSON string)
    # item = data[0]    
    # task_desc_str = item.get("task_desc", "")
    
    # # Now parse the JSON string
    # task_config = json.loads(task_desc_str)
    # print(f"task_config = {task_config}")
    # task_name = task_config["task_name"]
    # variation = task_config.get("var_num", 0)
    # jar_path = task_config.get("jar_path", "")
    # env = ScienceWorldEnv("", jar_path, envStepLimit=30)

    # simplificationStr = "easy"
    # env.load(task_name, variation, simplificationStr, generateGoldPath=True)

    # observation, info = env.reset()
    # print(f"observation = {observation}")
    # print(f"info = {info}")

    # print(f"task desc= {env.get_task_description()}")



    
