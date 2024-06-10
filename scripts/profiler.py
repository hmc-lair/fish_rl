from gymnasium.wrappers import TimeLimit, FlattenObservation
import matplotlib.pyplot as plt
from itertools import count
from fish_env import FishEnv
from tqdm import tqdm
import timeit
import numpy as np
from apf import init, update
from numba import jit

def time_episode(n):
    env = FlattenObservation(FishEnv("FollowFishSim"))
    def it():
        # Initialize the environment and get it's state
        env.reset()
        done = False
        while not done:
            action = 4
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    it()
    total_time = timeit.timeit(it, number=n)
    print(f"Total time: {total_time:.2f}, n: {n}, time per it: {total_time / n}")

def time_update(n):
    env = FlattenObservation(FishEnv("FollowFishSim"))
    env.reset()
    unwrapped = env.unwrapped
    def it():
        update(
            env.bounds,
            env.np_random,
            unwrapped._agent,
            unwrapped._fish,
            np.random.randint(0, 9),
            env.dt,
            env.attrs["dynamics"]
        )
    it()
    total_time = timeit.timeit(it, number=n)
    print(f"Total time: {total_time:.2f}, n: {n}, time per it: {total_time / n}")

if __name__ == "__main__":
    time_episode(10)
    time_update(10000)
