from gymnasium.wrappers import TimeLimit, FlattenObservation
from fish_env import FishEnv
import torch
from dqn import DQN
import argparse

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_episode(policy_net):
    env = FlattenObservation(TimeLimit(FishEnv(name="RealTest", render_mode="human"), max_episode_steps=1000))

    # Initialize the environment and get it's state
    cum_reward = 0
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        action = policy_net(state).max(1).indices.view(1, 1)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

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
    args = parser.parse_args()
    policy_net = torch.load(args.filepath)
    show_episode(policy_net)