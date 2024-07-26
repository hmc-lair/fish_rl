import argparse
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import matplotlib
import matplotlib.pyplot as plt
from fish_env import FishEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Check if running in IPython/Jupyter Notebook
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def safe_divide(numerator, denominator):
    """Safe division function to handle division by zero and NaN values."""
    denominator = np.where(denominator == 0, np.nan, denominator)
    result = np.divide(numerator, denominator)
    return np.nan_to_num(result)

def show_episode(env, model, render_mode):
    state = env.reset()
    done = False
    cum_reward = 0

    while not done:
        action, _states = model.predict(state)
        state, reward, done, _ = env.step(action)
        cum_reward += reward
        if render_mode == "human":
            env.render()  # Render the environment if in human mode
    
    env.close()
    return cum_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="DefaultSim", help="name of the fish env settings to use")
    parser.add_argument("-r", "--render_mode", type=str, default="human", help="how to render the environment")
    parser.add_argument("-i", "--input", type=str, required=True, help="input file to load the trained network from")
    args = parser.parse_args()

    # Set up the environment
    env = DummyVecEnv([lambda: FlattenObservation(FishEnv(name=args.name, render_mode=args.render_mode))])

    # Load the model
    model = A2C.load(args.input, env=env)

    # Test the model
    cum_reward = show_episode(env, model, args.render_mode)
    
    # Print the result
    print(f"Total cumulative reward: {cum_reward}")

    # If in IPython/Jupyter Notebook, display the plot
    if is_ipython:
        plt.show()
