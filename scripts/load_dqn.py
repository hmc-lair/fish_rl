from gymnasium.wrappers import TimeLimit, FlattenObservation
from fish_env import FishEnv
import torch
from dqn import DQN
import argparse

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_episode(policy_net, name, render_mode):
    env = FlattenObservation(FishEnv(name=name, render_mode=render_mode))

    # Initialize the environment and get it's state
    cum_reward = 0.0
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
    parser.add_argument('filepath')
    parser.add_argument("-n", "--name", type=str, default="DefaultSim", help="name of the fish env settings to use")
    parser.add_argument("-r", "--render_mode", type=str, default="human", help="how to render the environment")
    args = parser.parse_args()
    args = parser.parse_args()
    policy_net = torch.load(args.filepath)
    show_episode(policy_net, args.name, args.render_mode)