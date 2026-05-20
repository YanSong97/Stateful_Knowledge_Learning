import numpy as np
import random
from alfworld.agents.environment import get_environment
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List


class AlfWorldEnv_Wrapper:
    """
    Wrapper for AlfWorld environment following the same pattern as FrozenLake wrapper.
    """
    
    def __init__(self, env_config, **kwargs):
        """
        Initialize AlfWorld environment wrapper.
        
        Args:
            env_config: Configuration object with attributes:
                - config_path: Path to AlfWorld config YAML file
                - env_type: Type of environment ('AlfredTWEnv', 'AlfredThorEnv', or 'AlfredHybrid')
                - train_eval: 'train' or 'eval'
                - batch_size: Batch size for environment (default: 1)
                - random_seed: Random seed for reproducibility
                - max_steps: Maximum steps per episode
        """
        self.config = env_config
        
        # Load AlfWorld config
        # if hasattr(env_config, 'env_config_path') and env_config.env_config_path:
        # if kwargs.get('env_config_path', None) is not None:
        #     config_path = Path(kwargs['env_config_path'])
        # else:
        #     # Default to base_config.yaml in the same directory
        config_path = Path(__file__).parent / 'config' / 'base_config.yaml'
        
        self.alfworld_config = OmegaConf.load(config_path)
        
        # Override config values from env_config if provided
        if hasattr(env_config, 'env_type'):
            self.alfworld_config['env']['type'] = env_config.env_type
        self.env_type = self.alfworld_config['env']['type']
        
        # train_eval = getattr(env_config, 'train_eval', 'train')
        # batch_size = getattr(env_config, 'batch_size', 1)
        self.train_eval = kwargs.get("train_eval", 'eval_out_of_distribution')
        batch_size = 1
        
        # Setup environment
        self.alfred_env = get_environment(self.env_type)(self.alfworld_config, train_eval=self.train_eval)
        # _env = self.alfred_env.init_env(batch_size=batch_size)
        self.total_num_games = self.alfred_env.num_games
        
        self.max_steps = getattr(env_config, 'max_steps', 50)
        self.current_step = 0
        self.game_status = None
        self.last_reset_seed = None
        self.last_obs = None
        self.last_info = None
        
    @property
    def num_game(self):
        """Return number of available games/tasks."""
        # AlfWorld doesn't have a fixed number of games, return a large number
        # or get from dataset if available
        return self.total_num_games
    
    @property
    def legal_moves_list(self):
        """Return list of legal moves/commands."""
        if self.last_info is not None and 'admissible_commands' in self.last_info:
            return list(self.last_info['admissible_commands'][0]) if len(self.last_info['admissible_commands']) > 0 else []
        return []
    
    @property
    def legal_moves(self):
        """Return legal moves (same as legal_moves_list for compatibility)."""
        return self.legal_moves_list
    
    @property
    def legal_moves_string(self):
        """Return legal moves as a string."""
        moves = self.legal_moves_list
        go_action = []
        examine_action = []
        open_action = []
        take_action = []
        close_action = []
        put_action = []
        clean_action = []
        move_action = []
        help_action = []
        look_action = []
        inventory_action = []

        for m in moves:
            if m.startswith("go"):
                go_action.append(m)
            elif m.startswith("examine"):
                examine_action.append(m)
            elif m.startswith("open"):
                open_action.append(m)
            elif m.startswith("take"):
                take_action.append(m)
            elif m.startswith("close"):
                close_action.append(m)
            elif m.startswith("put"):
                put_action.append(m)
            elif m.startswith("clean"):
                clean_action.append(m)
            elif m.startswith("move"):
                move_action.append(m)
            elif m.startswith("help"):
                help_action.append(m)
            elif m.startswith("look"):
                look_action.append(m)
            elif m.startswith("inventory"):
                inventory_action.append(m)
            else:
                raise NotImplementedError(f"Action {m} is not supported.")

        final_move_string_template = """\
Go: [{go_action}];
Examine: [{examine_action}];
Open: [{open_action}];
Take: [{take_action}];
Close: [{close_action}];
Put: [{put_action}];
Clean: [{clean_action}];
Move: [{move_action}];
Help: [{help_action}];
Look: [{look_action}];
Inventory: [{inventory_action}];
"""
        final_move_string = final_move_string_template.format(
            go_action=", ".join(go_action), 
            examine_action=", ".join(examine_action), 
            open_action=", ".join(open_action), 
            take_action=", ".join(take_action),     
            close_action=", ".join(close_action),
            put_action=", ".join(put_action), 
            clean_action=", ".join(clean_action), 
            move_action=", ".join(move_action), 
            help_action=", ".join(help_action), 
            look_action=", ".join(look_action), 
            inventory_action=", ".join(inventory_action))

        return final_move_string if final_move_string else ""
    
    def check_success(self):
        """Check if the task was completed successfully."""
        if self.last_info is not None:
            # Check if done and score indicates success
            return self.last_info.get('won', False) if isinstance(self.last_info, dict) else False
        return False
    
    def check_falling(self):
        """Check if agent failed (not applicable for AlfWorld, but kept for compatibility)."""
        return False
    
    
    def reset(self, seed=None, specified_game_idx=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            specified_game_idx: Specific game/task index to load (if supported)
        
        Returns:
            observation: Current observation/state
        """
        self.game_idx = specified_game_idx
        if specified_game_idx is not None:
            specified_game_file = self.alfred_env.game_files[specified_game_idx]
            self.alfred_env.game_files = [specified_game_file]
            self.alfred_env.num_games = 1

        self.env = self.alfred_env.init_env(batch_size=1)

        # self.env.seed(specified_game_idx)
        
        # Reset environment
        # AlfWorld reset returns (obs, info) tuple
        obs, info = self.env.reset()
        
        self.last_obs = obs
        self.last_info = info
        self.current_step = 0
        self.game_status = None
        self.task_goal = obs[0].split("Your task is to:")[-1].strip()
        self.all_past_action = []
        
        return self.observe()
    
    
    def observe(self):
        """
        Get current observation.
        
        Returns:
            observation: Current observation/state as string
        """

        _obs = self.last_obs[0]
        if "Welcome to TextWorld, ALFRED!" in _obs:
            _obs = _obs.split("Welcome to TextWorld, ALFRED!")[1].strip()
        return _obs

        # if self.last_obs is not None:
        #     # AlfWorld returns observations as a list (batch), get first element
        #     # if isinstance(self.last_obs, list) and len(self.last_obs) > 0:
        #     #     return self.last_obs[0]
        #     return self.last_obs
        # return ""
    
    def reset_state(self, state_info: Dict[str, Any]):
        """
        Reset environment to a specific state.
        
        Args:
            state_info: Dictionary containing state information
        
        Returns:
            observation: Current observation
        """

        # reload dataset
        self.alfred_env = get_environment(self.env_type)(self.alfworld_config, train_eval=self.train_eval)

        # select the same game idx
        self.reset(specified_game_idx=state_info['game_idx'])

        # re-assign the game state
        # self.env.batch_env.last = state_info['env_last']
        # self.env.batch_env.envs[0]._wrapped_env.state = state_info['state']
        # self.env.obs = state_info['env_obs']
        # self.env.last_commands = state_info['env_last_commands']

        # self.current_step = state_info['current_step']
        # self.game_status = state_info['game_status']
        # self.last_obs = state_info['obs']
        # self.last_info = state_info['info']
        
        # return state_info['obs']

        for a in state_info['all_past_action']:
            _ = self.step(a)

        return self.observe()
    
    def get_key_stats(self):
        """
        Get key information for future possible retrace.
        
        Returns:
            Dictionary containing state information
        """
        return {
            "env_last": [i for i in self.env.batch_env.last],
            "env_obs": self.env.obs,
            "env_last_commands": self.env.last_commands,
            "state": self.env.batch_env.envs[0]._wrapped_env.state,
            "config": self.config,
            "game_idx": self.game_idx,
            "current_step": self.current_step,
            "game_status": self.game_status,
            "obs": self.last_obs,
            "info": self.last_info,
            "all_past_action": self.all_past_action,
        }
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (string command for AlfWorld)
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        self.current_step += 1
        
        # AlfWorld expects actions as a list (batch)
        if isinstance(action, str):
            actions = [action]
        elif isinstance(action, list):
            actions = action
        else:
            raise ValueError(f"Invalid action type: {type(action)}")
        
        assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"
        # Step environment
        obs, scores, dones, infos = self.env.step(actions)
        self.all_past_action.append(actions[0])
        
        # Update state
        self.last_obs = obs
        self.last_info = infos[0] if isinstance(infos, list) and len(infos) > 0 else infos
        
        # Extract values from batch (assuming batch_size=1)
        obs_value = obs[0] if isinstance(obs, list) and len(obs) > 0 else obs
        score_value = scores[0] if isinstance(scores, (list, np.ndarray, tuple)) and len(scores) > 0 else scores
        done_value = dones[0] if isinstance(dones, (list, np.ndarray, tuple)) and len(dones) > 0 else dones
        
        # Check if episode is done
        if done_value:
            self.game_status = score_value > 0  # Success if score > 0
        
        # Check max steps
        if not done_value and self.current_step >= self.max_steps:
            done_value = True
            self.game_status = False
        
        return self.observe(), score_value, done_value, {}
    
    def check_results(self):
        """
        Check game results.
        
        Returns:
            "win" if successful, "lose" otherwise
        """
        if self.game_status is None:
            # Try to infer from last info
            if self.last_info is not None:
                if isinstance(self.last_info, dict):
                    won = self.last_info.get('won', False)
                    return "win" if won else "lose"
        assert self.game_status is not None, "Game status is not set"
        return "win" if self.game_status else "lose"
    
    def render(self, mode='text'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        
        Returns:
            Rendered observation
        """
        return self.observe()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


class AsyncAlfWorldEnv_Wrapper:
    """
    Async wrapper for AlfWorld environment (for async operations if needed).
    """
    
    def __init__(self, env_config, **kwargs):
        """Initialize async AlfWorld environment wrapper."""
        self.config = env_config
        
        # Load AlfWorld config
        if hasattr(env_config, 'config_path') and env_config.config_path:
            config_path = Path(env_config.config_path)
        else:
            config_path = Path(__file__).parent / 'config' / 'base_config.yaml'
        
        self.alfworld_config = OmegaConf.load(config_path)
        
        if hasattr(env_config, 'env_type'):
            self.alfworld_config['env']['type'] = env_config.env_type
        env_type = self.alfworld_config['env']['type']
        
        train_eval = getattr(env_config, 'train_eval', 'train')
        batch_size = getattr(env_config, 'batch_size', 1)
        
        self.env = get_environment(env_type)(self.alfworld_config, train_eval=train_eval)
        self.env = self.env.init_env(batch_size=batch_size)
        
        seed = getattr(env_config, 'random_seed', None)
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
            np.random.seed(seed)
        
        self.last_reset_seed = None
        self.last_reset_game_idx = None
        self.last_obs = None
        self.last_info = None
    
    def reset(self, seed=None, specified_game_idx=None):
        """Reset the environment."""
        self.last_reset_seed = seed
        self.last_reset_game_idx = specified_game_idx
        
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
            np.random.seed(seed)
        
        obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        self.last_info = info
        return self.observe()
    
    def reset_last_seed(self):
        """Reset using the last seed."""
        return self.reset(seed=self.last_reset_seed)
    
    def reset_state(self, state_info: Dict[str, Any]):
        """Reset environment to a specific state."""
        if 'last_reset_seed' in state_info:
            self.reset(seed=state_info['last_reset_seed'])
        elif 'last_reset_game_idx' in state_info:
            self.reset(specified_game_idx=state_info['last_reset_game_idx'])
        else:
            self.reset()
        return self.observe()
    
    def get_key_stats(self):
        """Get key information for future possible retrace."""
        return {
            "config": self.config,
            "last_reset_seed": self.last_reset_seed,
            "last_reset_game_idx": self.last_reset_game_idx,
            "obs": self.last_obs,
            "info": self.last_info,
        }
    
    @property
    def legal_moves_list(self):
        """Return list of legal moves."""
        if self.last_info is not None and 'admissible_commands' in self.last_info:
            return list(self.last_info['admissible_commands'][0]) if len(self.last_info['admissible_commands']) > 0 else []
        return []
    
    @property
    def legal_moves(self):
        """Return legal moves."""
        return self.legal_moves_list
    
    @property
    def legal_moves_string(self):
        """Return legal moves as string."""
        moves = self.legal_moves_list
        return ",".join(moves) if moves else ""
    
    @property
    def num_game(self):
        """Return number of available games."""
        return getattr(self.config, 'num_games', 10000)
    
    async def step(self, action):
        """Execute one step asynchronously."""
        if isinstance(action, str):
            actions = [action]
        elif isinstance(action, list):
            actions = action
        else:
            raise ValueError(f"Invalid action type: {type(action)}")
        
        obs, scores, dones, infos = self.env.step(actions)
        self.last_obs = obs
        self.last_info = infos[0] if isinstance(infos, list) and len(infos) > 0 else infos
        
        obs_value = obs[0] if isinstance(obs, list) and len(obs) > 0 else obs
        score_value = scores[0] if isinstance(scores, (list, np.ndarray, tuple)) and len(scores) > 0 else scores
        done_value = dones[0] if isinstance(dones, (list, np.ndarray, tuple)) and len(dones) > 0 else dones
        
        return obs_value, score_value, done_value, {}
    
    def observe(self):
        """Get current observation."""
        if self.last_obs is not None:
            if isinstance(self.last_obs, list) and len(self.last_obs) > 0:
                return self.last_obs[0]
            return self.last_obs
        return ""
    
    def render(self, mode='text'):
        """Render the environment."""
        return self.observe()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


if __name__ == "__main__":
    # Example usage
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        config_path: str = str(Path(__file__).parent / 'config' / 'base_config.yaml')
        env_type: str = 'AlfredTWEnv'
        train_eval: str = 'train'
        batch_size: int = 1
        random_seed: int = 42
        max_steps: int = 50
    
    test_cfg = TestConfig()
    test_env = AlfWorldEnv_Wrapper(test_cfg)
    
    # Reset environment
    init_obs = test_env.reset(specified_game_idx=42)
    print("Initial observation:")
    print(init_obs)
    print(f"\nLegal moves: {test_env.legal_moves_list}")
    
    # Take a step
    if test_env.legal_moves_list:
        action = test_env.legal_moves_list[0]
        next_obs, reward, done, info = test_env.step(action)
        print(f"\nAfter action '{action}':")
        print(f"Reward: {reward}, Done: {done}")
        print(f"Observation: {next_obs[:200]}...")  # Print first 200 chars
    
    # Get key stats
    key_stats = test_env.get_key_stats()
    print(f"\nKey stats keys: {list(key_stats.keys())}")

