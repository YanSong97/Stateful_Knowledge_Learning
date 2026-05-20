import numpy as np
import random
from fractions import Fraction
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
# from src.envs.frozen_lake.gym_frozenlake.env import FrozenLakeEnv, FrozenLakeEnvConfig
from tasks.classic_games.frozen_lake.gym_frozenlake.env import FrozenLakeEnv, FrozenLakeEnvConfig


class FrozenLakeEnv_Wrapper(FrozenLakeEnv):
    def __init__(self, env_config, **kwargs):
        frozenlake_config = FrozenLakeEnvConfig()
        frozenlake_config.is_slippery = env_config.is_slippery
        frozenlake_config.map_seed = env_config.map_seed
        frozenlake_config.size = env_config.size
        frozenlake_config.success_rate = kwargs.get(
            'success_rate', getattr(env_config, 'success_rate', 1.0 / 3.0)
        )
        frozenlake_config.slip_prob_ratio = kwargs.get(
            'slip_prob_ratio', getattr(env_config, 'slip_prob_ratio', (1.0, 1.0))
        )
        frozenlake_config.num_map = getattr(env_config, 'num_map', 1024)
        frozenlake_config.game_obs_type = getattr(env_config, 'game_obs_type', "raw")
        frozenlake_config.max_steps = getattr(env_config, 'max_steps', 20)

        super(FrozenLakeEnv_Wrapper, self).__init__(frozenlake_config)

        self.config = frozenlake_config
        self.game_idx = env_config.map_seed

        self.game_obs_type = env_config.game_obs_type
        assert self.game_obs_type in ["raw", "coordinate", "combine", "full"], f"Invalid game obs type: {self.game_obs_type}"

        self.render_mode = 'rgb_array'

    def set_state(self, state=None, player=None):
        raise NotImplementedError

    def get_available_action(self):
        return
    
    @property
    def num_game(self):
        return self.config.num_map

    @property
    def legal_moves_list(self):
        return ["left", "down", "right", "up"]  #"1,2,3,4"

    @property
    def legal_moves(self):
        return ["left", "down", "right", "up"]

    @property
    def legal_moves_string(self):
        return "left,down,right,up"

    def check_success(self):
        return self.desc[self.player_pos] == b"G"

    def check_falling(self):
        return self.desc[self.player_pos] == b"H"

    def compute_distance2goal(self):
        current_pos = self.player_pos
        goal_pos = np.where(self.desc==b"G")
        goal_pos = (goal_pos[0].item(), goal_pos[1].item())
        distance = abs(goal_pos[0]-current_pos[0]) + abs(goal_pos[1]-current_pos[1])
        return distance

    def copy(self):
        return FrozenLakeEnv_Wrapper(env_config=self.config)

    def reset(self, seed=None, specified_game_idx=None):
        """
        specified game idx set the game map
        """
        self.last_reset_seed = seed
        self.game_idx = specified_game_idx

        if specified_game_idx is not None:
            game_idx = specified_game_idx
        else:
            game_idx = random.randint(0, self.num_game - 1)

        super().reset(game_idx)
        self.game_status = None
        self.max_step = self.config.max_steps
        self.current_step = 0

        return self.observe()

    def reset_last_seed(self):
        return super().reset(self.last_reset_seed)

    def observe(self):
        raw = self.render()
        if self.game_obs_type == "coordinate":
            return self._convert_raw_obs_to_coordinate_obs(raw)
        elif self.game_obs_type == "combine":
            coord = self._convert_raw_obs_to_coordinate_obs(raw)
            return f"{raw}\n{coord}"
        elif self.game_obs_type == "full":
            coord = self._convert_raw_obs_to_coordinate_obs(raw)
            fail_rate = self._get_fail_rate_each_move()
            fr_lines = "\n".join(f"{k}: {v}" for k, v in fail_rate.items())
            return f"{raw}\n{coord}\nHole fall probability per action due to slippery dynamics:\n{fr_lines}"
        return raw

    def _convert_raw_obs_to_coordinate_obs(self, raw_obs):
        """Parse text grid (GRID_LOOKUP) into row,col coordinates; origin top-left."""
        lines = [ln.rstrip() for ln in raw_obs.strip().split("\n") if ln.strip()]
        if not lines:
            return raw_obs
        nrows = len(lines)
        ncols = len(lines[0])
        player_on_goal = "√"  # grid_lookup[5]: player on goal

        player_pos = None
        goal_pos = None
        holes_set = set()

        for r, line in enumerate(lines):
            if len(line) != ncols:
                raise ValueError(
                    f"Inconsistent grid row width: expected {ncols}, got {len(line)} on row {r}"
                )
            for c, ch in enumerate(line):
                if ch in ("P", "X", player_on_goal):
                    player_pos = (r, c)
                if ch in ("G", player_on_goal):
                    goal_pos = (r, c)
                if ch in ("O", "X"):
                    holes_set.add((r, c))

        if player_pos is None or goal_pos is None:
            raise ValueError(
                f"Could not parse player or goal from observation:\n{raw_obs!r}"
            )

        holes_sorted = sorted(holes_set)
        holes_line = ", ".join(f"({r},{c})" for r, c in holes_sorted)

        parts = [
            f"Grid size: {nrows}x{ncols}",
            "",
            f"Player: ({player_pos[0]},{player_pos[1]})",
            f"Goal: ({goal_pos[0]},{goal_pos[1]})",
            "",
            f"Holes: {holes_line}",
            # "",
            # "Safe tiles: all other coordinates",
        ]
        return "\n".join(parts)

    @staticmethod
    def _prob_to_literal_str(p: float) -> str:
        """Render probability as ``0.0``, ``1.0``, or a reduced fraction like ``1/3``."""
        eps = 1e-9
        if p <= eps:
            return "0.0"
        if p >= 1.0 - eps:
            return "1.0"
        fr = Fraction(p).limit_denominator(1000)
        if fr.denominator == 1:
            return f"{int(fr.numerator)}.0"
        return f"{fr.numerator}/{fr.denominator}"

    def _hole_fall_probs_by_move(self):
        """P(fall into hole | action) per move name; floats for comparison."""
        s = int(self.s)
        row, col = self.player_pos
        if self.desc[row, col] in (b"G", b"H"):
            return {name: 0.0 for name in self.legal_moves}

        out = {}
        for i, name in enumerate(self.legal_moves):
            gym_a = self.action_map[i + 1]
            p_hole = 0.0
            for trans in self.P[s][gym_a]:
                prob, next_s, _reward, _terminated = trans
                nr = int(next_s) // self.ncol
                nc = int(next_s) % self.ncol
                if self.desc[nr, nc] == b"H":
                    p_hole += prob
            out[name] = p_hole
        return out

    def _get_fail_rate_each_move(self):
        """P(fall into hole | take action) at current state under env dynamics.

        Uses the same transition model as Gymnasium FrozenLakeEnv (``self.P``):
        when slippery, the intended direction has probability ``success_rate`` and
        the two perpendicular directions split the rest equally; when not slippery,
        the chosen direction is deterministic.

        Returns:
            dict[str, str]: keys ``left``, ``down``, ``right``, ``up`` (wrapper
            actions 1–4); values like ``\"0.0\"``, ``\"1.0\"``, ``\"1/3\"``, ``\"2/3\"``.
            On goal or hole tiles, all values are ``\"0.0\"``.
        """
        return {
            k: self._prob_to_literal_str(v)
            for k, v in self._hole_fall_probs_by_move().items()
        }

    def reset_state(self, state_info):

        self.reset(specified_game_idx=state_info['game_idx'])
        self.s = state_info['s']
        self.external_cnt = state_info['external_cnt']
        self.lastaction = state_info['lastaction']

        return self.observe()


    def get_key_stats(self):
        """
        get the key information for future possible retrace
        """

        ret = {
            "config": self.config,
            "game_idx": self.game_idx,
            "external_cnt": self.external_cnt,
            "s": self.s,
            "lastaction": self.lastaction,
        }
        return ret
    
    def step(self, action):
        """1,2,3,4"""
        if isinstance(action, str):
            processed_action = self.legal_moves.index(action.lower()) + 1
        
        elif isinstance(action, int):
            assert action in [1,2,3,4], f"Invalid action: {action}"
            processed_action = action
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        fall_probs = self._hole_fall_probs_by_move()
        chosen = self.legal_moves[processed_action - 1]
        p_chosen = fall_probs[chosen]
        min_p = min(fall_probs.values())
        good = 1 if abs(p_chosen - min_p) < 1e-9 else 0

        self.current_step += 1
        next_obs, reward, done, info = super().step(processed_action)
        if done:
            self.game_status = info['success']
        
        if not done and self.current_step >= self.max_step:
            done = True
            self.game_status = False

        out_info = dict(info)
        out_info["good"] = good
        return self.observe(), reward, done, out_info
    
    def check_results(self):
        assert self.game_status is not None, "Game status is not set" 
        if self.game_status:
            return "win"
        else:
            return "lose"


class AsyncFrozenLakeEnv_Wrapper(FrozenLakeEnv):
    def __init__(self, env_config, **kwargs):
        frozenlake_config = FrozenLakeEnvConfig()
        frozenlake_config.is_slippery = env_config.is_slippery
        frozenlake_config.map_seed = env_config.map_seed
        super(AsyncFrozenLakeEnv_Wrapper, self).__init__(frozenlake_config)

        self.config = frozenlake_config

    def reset(self, seed=None, specified_game_idx=None):
        self.last_reset_seed = seed
        self.last_reset_game_idx = specified_game_idx

        if specified_game_idx is not None:
            game_idx = specified_game_idx
        else:
            game_idx = random.randint(0, self.num_game - 1)

        return super().reset(game_idx)

    def reset_last_seed(self):
        return super().reset(self.last_reset_seed)
    
    def reset_state(self, state_info):

        self.reset(specified_game_idx=state_info['last_reset_game_idx'])
        self.s = state_info['s']
        self.external_cnt = state_info['external_cnt']
        self.lastaction = state_info['lastaction']
        return self.observe()

        # new_env = FrozenLakeEnv_Wrapper(state_info['config'])
        # new_env.reset(state_info['last_reset_seed'])
        # new_env.s = state_info['s']
        # new_env.external_cnt = state_info['external_cnt']
        # new_env.lastaction = state_info['lastaction']
        # return new_env
    
    def get_key_stats(self):
        """
        get the key information for future possible retrace
        """

        ret = {
            "config": self.config,
            "last_reset_seed": self.last_reset_seed,
            "last_reset_game_idx": self.last_reset_game_idx,
            "external_cnt": self.external_cnt,
            "s": self.s,
            "lastaction": self.lastaction,
            "FEN": self.observe()
        }
        return ret
    
    @property
    def legal_moves_list(self):
        return ["left", "down", "right", "up"]

    @property
    def legal_moves(self):
        return ["left", "down", "right", "up"]
    
    @property
    def legal_moves_string(self):
        return "left,down,right,up"

    @property
    def num_game(self):
        return 1024    # hard code for now

    def check_success(self):
        return self.desc[self.player_pos] == b"G"

    def check_falling(self):
        return self.desc[self.player_pos] == b"H"

    def compute_distance2goal(self):
        current_pos = self.player_pos
        goal_pos = np.where(self.desc==b"G")
        goal_pos = (goal_pos[0].item(), goal_pos[1].item())
        distance = abs(goal_pos[0]-current_pos[0]) + abs(goal_pos[1]-current_pos[1])
        return distance
    
    async def step(self, action):
        if isinstance(action, str):
            processed_action = self.legal_moves.index(action.lower()) + 1
        elif isinstance(action, int):
            assert action in [1,2,3,4], f"Invalid action: {action}"
            processed_action = action
        else:
            raise ValueError(f"Invalid action type: {type(action)}")
        return super().step(processed_action)
    
    def observe(self):
        return self.render()



if __name__ == "__main__":
    test_cfg = FrozenLakeEnvConfig()
    test_cfg.is_slippery = False
    test_cfg.max_steps = 20
    test_cfg.is_slippery = False
    test_cfg.size=5
    # test_cfg.slip_prob_ratio = (1., 1.)

    test_env = FrozenLakeEnv_Wrapper(test_cfg)

    init_obs = test_env.reset(specified_game_idx=44)
    print(init_obs, "\n")
    imshow(test_env.render())
    plt.show()
    next_obs = test_env.step("down")

    print(next_obs[0], "\n")

    # print(test_env.check_results())
    # # print(test_env.render())
    # # test_env.step("down")
    # # print(test_env.render())

    # key_stats = test_env.get_key_stats()
    # print(key_stats)
    # _new_env = FrozenLakeEnv_Wrapper(test_cfg)

    # _new_env.reset_state(key_stats)
    # print('new state = \n', _new_env.render())









