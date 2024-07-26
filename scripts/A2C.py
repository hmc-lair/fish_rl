import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, FlattenObservation
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from fish_env import FishEnv
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb
from stable_baselines3 import A2C  # Using the A2C model from stable-baselines3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

# Initialize wandb
wandb.init(project="fish-rl")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            episode_reward = sum(self.locals['infos'][i]['episode']['r'] for i in range(len(self.locals['infos'])) if 'episode' in self.locals['infos'][i])
            self.episode_rewards.append(episode_reward)
            wandb.log({"cumulative reward": episode_reward})
        return True

def plot_cum_rewards(show_result=False):
    plt.figure(1)
    cum_rewards_t = torch.tensor(episode_cum_rewards, dtype=torch.float64)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.plot(cum_rewards_t.numpy())
    if len(cum_rewards_t) >= 100:
        means = cum_rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def show_episode(env, model, render_mode):
    env = FlattenObservation(FishEnv(name=env, render_mode=render_mode))
    cum_reward = 0
    state, info = env.reset()
    while True:
        action, _states = model.predict(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        cum_reward += reward

        if done:
            break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_episodes", type=int, help="number of episodes to train for")
    parser.add_argument("-n", "--name", type=str, default="DefaultSim", help="name of the fish env settings to use")
    parser.add_argument("-r", "--render_mode", type=str, default="human", help="how to render the environment")
    parser.add_argument("-o", "--output", type=str, help="output file to save the trained network to")
    args = parser.parse_args()

    env = DummyVecEnv([lambda: FlattenObservation(FishEnv(name=args.name))])

    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    model = A2C("MlpPolicy", env, verbose=0)  # Set verbose to 0 to suppress logging

    episode_cum_rewards = []

    reward_logger = RewardLoggerCallback()

    # Create a tqdm progress bar
    for i in tqdm(range(args.num_episodes), desc="Training"):
        model.learn(total_timesteps=1000, callback=reward_logger)
    
    if args.output is not None:
        model.save(args.output)

    show_episode(args.name, model, args.render_mode)
