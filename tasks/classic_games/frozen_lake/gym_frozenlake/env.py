import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
import numpy as np
from tasks.classic_games.frozen_lake.gym_frozenlake.config import FrozenLakeEnvConfig
from tasks.classic_games.frozen_lake.gym_frozenlake.utils import generate_random_map, all_seed
from tasks.classic_games.frozen_lake.gym_frozenlake.base import BaseDiscreteActionEnv
import random

class FrozenLakeEnv(BaseDiscreteActionEnv, GymFrozenLakeEnv):
    def __init__(self, config: FrozenLakeEnvConfig = FrozenLakeEnvConfig()):
        # Using mappings directly from config
        self.config = config
        self.GRID_LOOKUP = config.grid_lookup
        self.ACTION_LOOKUP = config.action_lookup
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = config.render_mode
        self.action_map = config.action_map
        self.MAP_LOOKUP = config.map_lookup

        # option_seed = [2, 3, 6, 12, 57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]
        # option_seed = [257,58,59,60,61,62,63]
        # seed = random.choice(option_seed)
        seed=config.map_seed

        random_map = generate_random_map(size=config.size, p=config.p, seed=seed)
        # print(f"random_map = \n{random_map}")

        # random_map=["FGHF", "FFFH", "SHFF", "FFFF"]

        BaseDiscreteActionEnv.__init__(self)
        GymFrozenLakeEnv.__init__(
            self,
            desc=random_map,
            is_slippery=config.is_slippery,
            render_mode=config.render_mode,
            slip_prob_ratio=config.slip_prob_ratio
        )
        self.max_episode = config.max_episode

    def reset(self, seed=None):
        # option_seed = [2, 3, 6]
        # seed = random.choice(option_seed)
        # seed = 2

        self.external_cnt = 0
        # try:
        with all_seed(seed):
            self.config.map_seed = seed
            self.__init__(self.config)
            GymFrozenLakeEnv.reset(self, seed=seed)
            return self.render()
        # except (RuntimeError, RuntimeWarning) as e:
        #     next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
        #     return self.reset(next_seed)
    
    def step(self, action: int):
        prev_pos = int(self.s)
        _, reward, done, _, _ = GymFrozenLakeEnv.step(self, self.action_map[action])
        next_obs = self.render()
        info = {"action_is_effective": prev_pos != int(self.s), "action_is_valid": True, "success": self.desc[self.player_pos] == b"G"}

        self.external_cnt += 1
        if self.external_cnt >= self.max_episode:
            done = True

        return next_obs, reward, done, info
     
    def render(self):
        if self.render_mode == 'text':
            room = self.desc.copy()
            # replace the position of start 'S' with 'F', mark the position of the player as 'p'.
            room = np.where(room == b'S', b'F', room)
            room[self.player_pos] = b'P'
            room = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room)
            # add player in hole or player on goal
            room[self.player_pos] = 4 if self.desc[self.player_pos] == b'H' else 5 if self.desc[self.player_pos] == b'G' else 0

            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room)
        elif self.render_mode == 'rgb_array':
            return self._render_gui('rgb_array')
        else:
            raise ValueError(f"Invalid mode: {self.render_mode}")
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])

    @property
    def player_pos(self):
        return (self.s // self.ncol, self.s % self.ncol) # (row, col)

    def close(self):
        self.render_cache = None
        super(FrozenLakeEnv, self).close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    config = FrozenLakeEnvConfig(size=4, p=0.8, is_slippery=False)
    env = FrozenLakeEnv(config)
    print(env.reset(seed=123))
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        # assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    np_img = env.render('rgb_array')
    # save the image
    # plt.imsave('frozen_lake.png', np_img)
    # seed 2, 3, 6
