import numpy as np
from alfworld.agents.environment import get_environment
from omegaconf import OmegaConf
from pathlib import Path





"""
Available commands:
  look:                             look around your current location
  inventory:                        check your current inventory
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):  place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object
"""



# base_config: 
# eval_out_of_distribution: total load 341, num_game 134
# train: total load 8810, num_game 3553
# eval_in_distribution: total load 494, num_game 140

# eval_config:
# eval_out_of_distribution:  total 341, num_game 18
# train: total load 8810, num_game 308
# eval_in_distribution: total 494, num_game 13



# load config
config_path = Path(__file__).parent / 'config' / 'base_config.yaml'
config = OmegaConf.load(config_path)
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
alfred_env = get_environment(env_type)(config, train_eval='eval_in_distribution')    # load from local file ['train', 'eval_in_distribution', 'eval_out_of_distribution']
env = alfred_env.init_env(batch_size=1)
map_seed = 1

print(f"num games = {alfred_env.num_games}")
alfred_env.game_files = [alfred_env.game_files[0]]
alfred_env.num_games = len(alfred_env.game_files)

# env.batch_env.envs[0]._wrapped_env._game_file
# env.batch_env.last
env.seed(map_seed)

# interact
obs, info = env.reset()
print(f"obs = {obs}")
# print(f"info = {info}")
for _ in range(3):
  admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
  # print(f"admissible_commands = {admissible_commands}")
  random_actions = [np.random.choice(admissible_commands[0])]
  # print(f"random_actions = {random_actions}")
  obs, scores, dones, infos = env.step(random_actions)

print(f"state = {env.render()}")


# print(f"infos = {infos}")
env_last = [i for i in env.batch_env.last]
env_state = env.batch_env.envs[0]._wrapped_env.state
env_obs = env.obs
env_last_commands = env.last_commands

new_env = get_environment(env_type)(config, train_eval='train')
new_env = new_env.init_env(batch_size=1)
new_env.seed(map_seed)
new_obs, new_info = new_env.reset()
# append last
new_env.batch_env.last = env_last
new_env.batch_env.envs[0]._wrapped_env.state = env_state
new_env.obs = env_obs
new_env.last_commands = env_last_commands

print(f"new state = {new_env.render()}")
import pdb; pdb.set_trace()

print(1)



# while True:
#     # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
#     admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
#     random_actions = [np.random.choice(admissible_commands[0])]

#     # step
#     obs, scores, dones, infos = env.step(random_actions)
#     print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))