import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from apf import init, update, lines_from_bounds
import pdb

class FishEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, max_v=5, max_omega=2*np.pi, render_mode=None, n_fish=10, bounds=np.array([[0, 0], [10, 10]]), seed=None):
        self.window_size = 1024  # The size of the PyGame window

        # Bounds of the simulation
        # np.array([
        #     [minx, miny],
        #     [maxx, maxy]
        # ])
        self.bounds = bounds
        (self.minx, self.miny), (self.maxx, self.maxy) = self.bounds

        # Number of fish to simulte
        self.n_fish = n_fish

        # The observation space gives the state of the agent and the fish
        # (x, y, theta, v, omega)
        low = np.array([self.minx, self.miny, 0,     -np.inf, -np.inf])
        high = np.array([self.maxx, self.maxy, np.pi,  np.inf,  np.inf])
        self.observation_space = spaces.Box(
            np.stack([low] * (1 + n_fish)),
            np.stack([high] * (1 + n_fish))
        )

        # An action is a linear and angular velocity command
        self.max_v = max_v
        self.max_omega = max_omega
        # self.action_space = spaces.Tuple([
        #     spaces.Discrete(3, start=-1),
        #     spaces.Discrete(3, start=-1)
        # ])
        self.action_space = spaces.Discrete(9)
        # self.action_space = spaces.Box(
        #     np.array([-self.max_v, -self.max_omega]),
        #     np.array([ self.max_v,  self.max_omega])
        # )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # Negative reward is given at each time step, with the amount of negative reward depending on a TBD clustering metric
        self.reward_range = (-np.inf, 0)

        # self.dt = 0.05
        self.dt = 1 / self.metadata["render_fps"]

        # # Set up initial state
        # self.reset(seed=seed)

    def _get_obs(self):
        return np.vstack([self._agent, self._fish])

    def _get_info(self):
        # TODO: provide useful diagnostics here
        return {}

    def _action_to_vels(self, action):
        return np.array([self.max_v, self.max_omega]) * np.array([
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1]
        ])[action]

    def step(self, action):
        self._agent, self._fish = update(self.bounds, self.np_random, self._agent, self._fish, self._action_to_vels(action), self.dt)

        # Calculate reward using the fish covariance
        # The eigenvalues `l1` and `l2` of the covariance matrix are calculated
        # These are the axes of the covariance ellipse
        # https://cookierobotics.com/007/
        cov = np.cov(self._fish[:, :2].T)
        l1 = (cov[0][0] + cov[1][1]) / 2 + np.sqrt(np.square((cov[0][0] - cov[1][1]) / 2) + np.square(cov[0][1]))
        l2 = (cov[0][0] + cov[1][1]) / 2 - np.sqrt(np.square((cov[0][0] - cov[1][1]) / 2) + np.square(cov[0][1]))
        reward = -np.sqrt(l1) - np.sqrt(l2)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the states of the robot and fish
        self._agent, self._fish = init(self.bounds, self.n_fish, self.np_random)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _scale_to_window(self, x):
        m = int(self.window_size * 0.05) * 2
        w = self.window_size - m
        return w * x / self.bounds[1][0]

    def _scale_to_bounds(self, x):
        m = int(self.window_size * 0.05) * 2
        w = self.window_size - m
        return self.bounds[1][0] * x / w

    def _to_window_coords(self, coords):
        m = int(self.window_size * 0.05) * 2
        w = self.window_size - m
        x, y = coords
        x = w * (x - self.bounds[0][0]) / self.bounds[1][0] + m / 2
        y = w * (y - self.bounds[0][1]) / self.bounds[1][1] + m / 2
        return np.stack([x, self.window_size - y]).T

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        r = self._scale_to_bounds(30)
        # Draw the fish
        for fish in self._fish:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                self._to_window_coords(fish[:2]),
                self.window_size / 100
            )
            end = fish[:2] + r * np.array([np.cos(fish[2]), np.sin(fish[2])])
            pygame.draw.line(
                canvas,
                (255, 0, 0),
                self._to_window_coords(fish[:2]),
                self._to_window_coords(end),
                width=2
            )
        
        cov = np.cov(self._fish[:, :2].T)
        l1 = (cov[0][0] + cov[1][1]) / 2 + np.sqrt(np.square((cov[0][0] - cov[1][1]) / 2) + np.square(cov[0][1]))
        l2 = (cov[0][0] + cov[1][1]) / 2 - np.sqrt(np.square((cov[0][0] - cov[1][1]) / 2) + np.square(cov[0][1]))
        if cov[0][1] == 0:
            if cov[0][0] >= cov[1][1]:
                theta = 0
            else:
                theta = np.pi / 2
        else:
            theta = np.arctan2(l1 - cov[0][0], cov[0][1])
        x = np.average(self._fish[:, 0])
        y = np.average(self._fish[:, 1])
        w = 2 * np.sqrt(l1)  # 1st standard deviation
        h = 2 * np.sqrt(l2)
        target_rect = pygame.Rect([
            *self._to_window_coords(np.array([x-w/2, y+h/2])),
            *self._scale_to_window(np.array([w, h]))
        ])
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(
            shape_surf,
            (255, 0, 0),
            [0, 0, *target_rect.size],
            3
        )
        pygame.draw.line(
            shape_surf,
            (255, 0, 0),
            [target_rect.size[0] / 2, 0],
            [target_rect.size[0] / 2, target_rect.size[1]],
            3
        )
        pygame.draw.line(
            shape_surf,
            (255, 0, 0),
            [0, target_rect.size[1] / 2],
            [target_rect.size[0], target_rect.size[1] / 2],
            3
        )
        rotated_surf = pygame.transform.rotate(shape_surf, theta * 180 / np.pi)
        canvas.blit(rotated_surf, rotated_surf.get_rect(center=target_rect.center))

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._to_window_coords(self._agent[:2]),
            self.window_size / 90
        )
        end = self._agent[:2] + r * np.array([np.cos(self._agent[2]), np.sin(self._agent[2])])
        pygame.draw.line(
            canvas,
            (0, 0, 255),
            self._to_window_coords(self._agent[:2]),
            self._to_window_coords(end),
            width=2
        )

        # Draw the bounds
        for line in lines_from_bounds(self.bounds):
            pygame.draw.line(
                canvas,
                0,
                self._to_window_coords(line[0]),
                self._to_window_coords(line[1]),
                width=int(self.window_size / 100)
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpace(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    a = np.array([0, 0], dtype=np.int32)
    a_map = {-1: {-1: 0, 0: 1, 1: 2}, 0: {-1: 3, 0: 4, 1: 5}, 1: {-1: 6, 0: 7, 1: 8}}
    env = gym.wrappers.TimeLimit(FishEnv(seed=42, render_mode="human"), max_episode_steps=1000)

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    a[1] = 1
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    a[1] = -1
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    a[0] = 1
                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    a[0] = -1
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    a[1] = 0
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    a[1] = 0
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    a[0] = 0
                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    a[0] = 0

            if event.type == pygame.QUIT:
                quit = True

    quit = False
    # while not quit:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        register_input()
        s, r, terminated, truncated, info = env.step(a_map[a[0]][a[1]])
        total_reward += r
        if steps % 200 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated or restart or quit:
            break
    env.close()