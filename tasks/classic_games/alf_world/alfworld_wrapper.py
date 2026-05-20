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
        if hasattr(env_config, 'config_path') and env_config.config_path:
            config_path = Path(env_config.config_path)
        else:
            # Default to base_config.yaml in the same directory
            config_path = Path(__file__).parent / 'config' / 'base_config.yaml'
        
        self.alfworld_config = OmegaConf.load(config_path)
        
        # Override config values from env_config if provided
        if hasattr(env_config, 'env_type'):
            self.alfworld_config['env']['type'] = env_config.env_type
        env_type = self.alfworld_config['env']['type']
        
        train_eval = getattr(env_config, 'train_eval', 'train')
        batch_size = getattr(env_config, 'batch_size', 1)
        
        # Setup environment
        self.env = get_environment(env_type)(self.alfworld_config, train_eval=train_eval)
        self.env = self.env.init_env(batch_size=batch_size)
        
        # Set seed if provided
        seed = getattr(env_config, 'random_seed', None)
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
            np.random.seed(seed)
            self.default_seed = seed
        else:
            self.default_seed = None
        
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
        return getattr(self.config, 'num_games', 10000)
    
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
        return ",".join(moves) if moves else ""
    
    def check_success(self):
        """Check if the task was completed successfully."""
        if self.last_info is not None:
            # Check if done and score indicates success
            return self.last_info.get('won', False) if isinstance(self.last_info, dict) else False
        return False
    
    def check_falling(self):
        """Check if agent failed (not applicable for AlfWorld, but kept for compatibility)."""
        return False
    
    def compute_distance2goal(self):
        """Compute distance to goal (not directly applicable for AlfWorld)."""
        # AlfWorld doesn't have a simple distance metric
        return 0
    
    def copy(self):
        """Create a copy of the environment."""
        return AlfWorldEnv_Wrapper(env_config=self.config)
    
    def reset(self, seed=None, specified_game_idx=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            specified_game_idx: Specific game/task index to load (if supported)
        
        Returns:
            observation: Current observation/state
        """
        self.last_reset_seed = seed if seed is not None else self.default_seed
        
        # Set seed if provided
        if self.last_reset_seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(self.last_reset_seed)
            np.random.seed(self.last_reset_seed)
        
        # Reset environment
        # AlfWorld reset returns (obs, info) tuple
        obs, info = self.env.reset(seed=self.last_reset_seed if self.last_reset_seed is not None else None)
        
        self.last_obs = obs
        self.last_info = info
        self.current_step = 0
        self.game_status = None
        
        return self.observe()
    
    def reset_last_seed(self):
        """Reset using the last seed."""
        return self.reset(seed=self.last_reset_seed)
    
    def observe(self):
        """
        Get current observation.
        
        Returns:
            observation: Current observation/state as string
        """
        if self.last_obs is not None:
            # AlfWorld returns observations as a list (batch), get first element
            if isinstance(self.last_obs, list) and len(self.last_obs) > 0:
                return self.last_obs[0]
            return self.last_obs
        return ""
    
    def reset_state(self, state_info: Dict[str, Any]):
        """
        Reset environment to a specific state.
        
        Args:
            state_info: Dictionary containing state information
        
        Returns:
            observation: Current observation
        """
        # Restore from saved state
        if 'last_reset_seed' in state_info:
            self.reset(seed=state_info['last_reset_seed'])
        elif 'game_idx' in state_info:
            self.reset(specified_game_idx=state_info['game_idx'])
        else:
            self.reset()
        
        # Restore additional state if available
        if 'current_step' in state_info:
            self.current_step = state_info['current_step']
        if 'game_status' in state_info:
            self.game_status = state_info['game_status']
        
        return self.observe()
    
    def get_key_stats(self):
        """
        Get key information for future possible retrace.
        
        Returns:
            Dictionary containing state information
        """
        return {
            "config": self.config,
            "last_reset_seed": self.last_reset_seed,
            "current_step": self.current_step,
            "game_status": self.game_status,
            "obs": self.last_obs,
            "info": self.last_info,
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
        
        # Step environment
        obs, scores, dones, infos = self.env.step(actions)
        
        # Update state
        self.last_obs = obs
        self.last_info = infos[0] if isinstance(infos, list) and len(infos) > 0 else infos
        
        # Extract values from batch (assuming batch_size=1)
        obs_value = obs[0] if isinstance(obs, list) and len(obs) > 0 else obs
        score_value = scores[0] if isinstance(scores, (list, np.ndarray)) and len(scores) > 0 else scores
        done_value = dones[0] if isinstance(dones, (list, np.ndarray)) and len(dones) > 0 else dones
        
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
        score_value = scores[0] if isinstance(scores, (list, np.ndarray)) and len(scores) > 0 else scores
        done_value = dones[0] if isinstance(dones, (list, np.ndarray)) and len(dones) > 0 else dones
        
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
    init_obs = test_env.reset(seed=42)
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


