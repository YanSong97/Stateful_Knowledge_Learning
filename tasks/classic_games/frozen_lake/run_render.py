import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', desc=["FGHF", "FFFH", "SHFF", "FFFF"], map_name="4x4", is_slippery=True, render_mode="rgb_array")

observation, info = env.reset()

render_obs = env.render()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    render_obs = env.render()

    episode_over = terminated or truncated

env.close()