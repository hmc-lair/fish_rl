# Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

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

BATCH_SIZE = 128    # number of transitions sampled from the replay buffer
GAMMA = 0.99        # discount factor
EPS_START = 0.9     # starting value of epsilon
EPS_END = 0.05      # final value of epsilon
EPS_DECAY = 1000    # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005         # update rate of the target network
LR = 1e-4           # learning rate of the ``AdamW`` optimizer

# A named tuple representing a single transition in our environment. It essentially maps (state, action) pairs to their (next_state, reward) result, with the state being the screen difference image as described later on.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    '''
    A cyclic buffer of bounded size that holds the transitions observed recently. It also implements a `.sample()` method for selecting a random batch of transitions for training.
    '''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # self.layers = nn.ModuleList([
        #    nn.Linear(n_observations, 32).double(),
        #    nn.Linear(32, 64).double(),
        #    nn.Linear(64, 128).double(),
        #    nn.Linear(128, 128).double(),
        #    nn.Linear(128, 64).double(),
        #    nn.Linear(64, 32).double(),
        #    nn.Linear(32, n_actions).double()
        # ])
        self.layers = nn.ModuleList([
             nn.Linear(n_observations, 32).double(),
             nn.Linear(32, 32).double(),
             nn.Linear(32, n_actions).double()
         ])
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = F.relu(self.layer1(x))
        # return F.relu(self.layer2(x))
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x
        # return self.layers(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
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
    # Take 100 episode averages and plot them too
    if len(cum_rewards_t) >= 100:
        means = cum_rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)  # pause a bit so that plots are updated
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
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float64)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion =  nn.MSELoss(size_average=None, reduce=None, reduction='mean') #nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def show_episode(name, render_mode):
    env = FlattenObservation(FishEnv(name=name, render_mode=render_mode))

    # Initialize the environment and get it's state
    cum_reward = 0
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)
    while True:
        action = policy_net(state).max(1).indices.view(1, 1)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)

        # Move to the next state
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

    # set up matplotlib
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        print("Is ipython")
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = spaces.utils.flatdim(env.action_space)
    # Get the number of state observations
    # state, info = env.reset()
    n_observations = spaces.utils.flatdim(env.observation_space)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    episode_cum_rewards = []

    steps_done = 0

    if torch.cuda.is_available():
        print("Cuda is available")
    else:
        print("Cuda is not available")
    num_episodes = args.num_episodes

    wandb.init(
        # set the wandb project where this run will be logged
        project="fish-rl",

        # track hyperparameters and run metadata
        config={
            "environment": args.name,
            "output": os.path.split(args.output),
            "algorithm": "dqn",
            "episodes": num_episodes,
        }
    )

    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get it's state
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

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            cum_reward += reward

            if done:
                wandb.log({"cumulative reward": cum_reward})
                break
    
    if args.output is not None:
        torch.save(policy_net, args.output)

