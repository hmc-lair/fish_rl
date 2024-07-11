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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128).double(),
            nn.ReLU(),
            nn.Linear(128, 128).double(),
            nn.ReLU(),
            nn.Linear(128, n_actions).double()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.layers(x)
        x = torch.clamp(x, min=-10, max=10)  # Clamping to avoid extreme values
        return self.softmax(x)


class Critic(nn.Module):
    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128).double(),
            nn.ReLU(),
            nn.Linear(128, 128).double(),
            nn.ReLU(),
            nn.Linear(128, 1).double()
        )
    
    def forward(self, x):
        return self.layers(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Ensure state is free of NaNs and Infs
            if torch.isnan(state).any() or torch.isinf(state).any():
                raise ValueError("State contains NaN or Inf values")
            return actor_net(state).multinomial(1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

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

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Critic optimization
    state_values = critic_net(state_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float64)
    with torch.no_grad():
        next_state_values[non_final_mask] = critic_net(non_final_next_states).squeeze()
    expected_state_values = reward_batch + (next_state_values * GAMMA)

    critic_loss = F.mse_loss(state_values, expected_state_values.unsqueeze(1))
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor optimization
    log_probs = torch.log(actor_net(state_batch).gather(1, action_batch))
    advantages = (expected_state_values.unsqueeze(1) - state_values).detach()
    actor_loss = -(log_probs * advantages).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

def show_episode(name, render_mode):
    env = FlattenObservation(FishEnv(name=name, render_mode=render_mode))
    cum_reward = 0
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)
    while True:
        action = actor_net(state).multinomial(1)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)

        state = next_state
        cum_reward += reward

        if done:
            print("Cumulative reward:", cum_reward)
            break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_episodes", type=int, help="number of episodes to train for")
    parser.add_argument("-n", "--name", type=str, default="DefaultSim", help="name of the fish env settings to use")
    parser.add_argument("-r", "--render_mode", type=str, default="human", help="how to render the environment")
    parser.add_argument("-o", "--output", type=str, help="output file to save the trained network to")
    args = parser.parse_args()

    env = FlattenObservation(FishEnv(name=args.name))

    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = spaces.utils.flatdim(env.action_space)
    n_observations = spaces.utils.flatdim(env.observation_space)

    actor_net = Actor(n_observations, n_actions).to(device)
    critic_net = Critic(n_observations).to(device)

    actor_optimizer = optim.AdamW(actor_net.parameters(), lr=LR)
    critic_optimizer = optim.AdamW(critic_net.parameters(), lr=LR)
    memory = ReplayMemory(10000)

    episode_cum_rewards = []

    steps_done = 0

    if torch.cuda.is_available():
        print("Cuda is available")
    else:
        print("Cuda is not available")
    num_episodes = args.num_episodes

    wandb.init(
        project="fish-rl",
        config={
            "episodes": num_episodes,
        }
    )

    for i_episode in tqdm(range(num_episodes)):
        cum_reward = 0.0
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            cum_reward += reward

            if done:
                wandb.log({"cumulative reward": cum_reward})
                break
    
    if args.output is not None:
        torch.save(actor_net, args.output)
