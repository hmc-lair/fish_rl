import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from apf import init, update, lines_from_bounds
import pdb
import yaml
from testbed.cyberbot.rl_controller import RLController
import time
import argparse

class FishEnv(gym.Env):
    metadata = {"render_modes": ["camera", "threshold", "human", "rgb_array"], "render_fps": 20}
    fish_settings_yaml = "../config/fish_env_settings.yaml"

    def __init__(self, name="DefaultSim", render_mode=None, seed=None):
        self.window_size = 2048  # The size of the larger side of the PyGame window
        self.window_margin = int(self.window_size * (0.1 / 2))  # Width of margin. Both the left and right, and top and bottom margins are this width, so the total amount of margin on both axes is double this
        self.attrs = self._parse_settings(name)
        self.title = self.attrs.get("title", name)
        self.seed = seed

        # Check sim vs real and rendering mode
        assert self.attrs["type"] in ["sim", "real"]
        self.type = self.attrs["type"]
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode == "camera": assert self.type == "real"
        self.render_mode = render_mode

        if self.type == "sim":
            self._rl_controller = None
        else:
            self._rl_controller = RLController(
                self.attrs["transmitter_port"],
                self.attrs["camera_port"],
                self.attrs["robot_params_path"],
                self.attrs["camera_calibration_path"],
                self.attrs["object_detection_path"],
                self.attrs["n_fish"],
                render_mode=render_mode,
                use_pygame=True
            )

        if(self.seed == None):
            self.seed = self.attrs["seed"]

        # Bounds of the simulation
        # np.array([
        #     [minx, miny],
        #     [maxx, maxy]
        # ])
        if self.type == "sim":
            self.bounds =np.array([
                [self.attrs["bounds"]["min_x"], self.attrs["bounds"]["min_y"]],
                [self.attrs["bounds"]["max_x"], self.attrs["bounds"]["max_y"]]
            ])
        else:
            camera_bounds = self._rl_controller._video._bounds
            meters_per_pixel = self._rl_controller._video._calibration_params["METERS_PER_PIXEL"]
            self.bounds = np.array([
                [0, 0],
                (camera_bounds[1] - camera_bounds[0]) * meters_per_pixel
            ])
        (self.minx, self.miny), (self.maxx, self.maxy) = self.bounds

        # Get window width and height from window size, preserving the bounds ratio
        self.width = self.maxx - self.minx
        self.height = self.maxy - self.miny
        self.window_width, self.window_height = (self.window_size - 2 * self.window_margin) * np.array([
            [1, self.height / self.width],
            [self.width / self.height, 1]
        ])[np.argmin([self.height / self.width, self.width / self.height])] + 2 * self.window_margin

        # Number of fish to simulate
        self.n_fish = self.attrs["n_fish"]

        # The observation space gives the state of the agent and the fish
        # (x, y, theta, v, omega)
        low = np.array([self.minx, self.miny, 0,     -np.inf, -np.inf])
        high = np.array([self.maxx, self.maxy, np.pi,  np.inf,  np.inf])
        self.observation_space = spaces.Box(
            np.stack([low] * (1 + self.n_fish)),
            np.stack([high] * (1 + self.n_fish))
        )

        # An action is a linear and angular velocity command
        if self.type == "sim":
            self.max_v = self.attrs["dynamics"]["max_lin_v"]
            self.max_omega = self.attrs["dynamics"]["max_ang_v"]
        else:
            self.max_v = self.attrs["dynamics"].get("max_lin_v", self._rl_controller._params["V_MAX"])
            self.max_omega = self.attrs["dynamics"].get("max_ang_v", self._rl_controller._params["OMEGA_MAX"])

        # self.action_space = spaces.Discrete(9)
        self.action_space = spaces.Discrete(6)

        # Implement a time limit
        self._max_episode_steps = self.attrs.get("max_episode_steps", None)
        self._elapsed_steps = None

        # Create the reward function using the settings in the yaml file
        if "reward" not in self.attrs or self.attrs["reward"] is None:
            if "reward" not in self.attrs:
                print("Warning: no reward type specified. Defaulting to no reward")
            self._get_reward = lambda agent, fish: (0, False)
        else:
            reward_params = self.attrs["reward"]
            assert "type" in reward_params
            if reward_params["type"] in ["distance", "continious", "radius", "positional", "direction", "combo", "comboPosDis"]:
                assert "target" in reward_params
                if reward_params["type"] == "radius":
                    assert "radius" in reward_params
                    radius = reward_params["radius"]

                if reward_params["target"] == "fish":
                    assert self.n_fish == 1
                    if reward_params["type"] == "distance":
                        self._get_reward = lambda agent, fish: (-np.linalg.norm(agent[:2] - fish[0][:2]), False)  # Distance from fish
                    elif reward_params["type"] == "shifted_distance":
                        diagonal = np.sqrt(np.sum(np.square(self.bounds[1] - self.bounds[0])))
                        self._get_reward = lambda agent, fish: (diagonal-np.linalg.norm(agent[:2] - target), False)  # Negative distance from target point shifted up to always be positive
                    else:
                        self._get_reward = lambda agent, fish: (0, False) if np.linalg.norm(agent[:2] - fish[0][:2]) > radius else (1, True)  # Radius from fish
                elif isinstance(reward_params["target"], list):
                    target = np.array(reward_params["target"])
                    assert target.shape == (2,)
                    if reward_params["type"] == "distance":
                        self._get_reward = lambda agent, fish: (-np.linalg.norm(agent[:2] - target), False)  # Negative distance from target point
                    elif reward_params["type"] == "shifted_distance":
                        diagonal = np.sqrt(np.sum(np.square(self.bounds[1] - self.bounds[0])))
                        self._get_reward = lambda agent, fish: (diagonal-np.linalg.norm(agent[:2] - target), False)  # Negative distance from target point shifted up to always be positive
                    elif reward_params["type"] == "continious":
                        self._get_reward = lambda agent, fish: (1 / (1 + np.linalg.norm(fish[0][:2] - target)), False)
                    elif reward_params["type"] == "positional":
                        self._get_reward =  lambda agent, fish: ((1 + np.dot((target - fish[0][:2]) / np.linalg.norm(target - fish[0][:2]),(fish[0][:2] - agent[:2]) / np.linalg.norm(fish[0][:2] - agent[:2]))) / 2, False)
                    elif reward_params["type"] == "direction":
                        self._get_reward = lambda agent, fish: (1 if (fish[0][:2] - agent[:2])[0] < 0 else -1, False)
                    elif reward_params["type"] == "combo":
                        self._get_reward = lambda agent, fish: ( \
                            (1 / (1 + np.linalg.norm(fish[0][:2] - target)) + (1 if (fish[0][:2] - agent[:2])[0] < 0 else -1)) \
                        , False)
                    elif reward_params["type"] == "comboPosDis":
                        self._get_reward = lambda agent, fish: ((1 + np.dot((target - fish[0][:2]) / np.linalg.norm(target - fish[0][:2]),(fish[0][:2] - agent[:2]) / np.linalg.norm(fish[0][:2] - agent[:2]))) / 2 + (1 if (fish[0][:2] - agent[:2])[0] < 0 else -1), False)
                    else:
                        self._get_reward = lambda agent, fish: (0, False) if np.linalg.norm(agent[:2] - target) > radius else (1, True)  # Radius from fish
                else:
                    raise ValueError(f"Unknown target: {reward_params['target']}")
            elif reward_params["type"] == "count":
                assert "bounds" in reward_params
                assert self._max_episode_steps is not None
                w, h = self.maxx - self.minx, self.maxy - self.miny
                herding_bounds = np.array([
                    [reward_params["bounds"]["minx"], reward_params["bounds"]["miny"]],
                    [reward_params["bounds"]["maxx"], reward_params["bounds"]["maxy"]]
                ]) * np.array([w, h]) + np.array([self.minx, self.miny])
                self._get_reward = lambda agent, fish: (np.sum(
                    (herding_bounds[0, 0] <= fish[:, 0]) & (fish[:, 0] <= herding_bounds[1, 0]) & \
                    (herding_bounds[0, 1] <= fish[:, 1]) & (fish[:, 1] <= herding_bounds[1, 1])
                ), False) if self._elapsed_steps == self._max_episode_steps - 1 else (0, False)
            else:
                raise ValueError(f"Unknown reward type: {reward_params['type']}")

            if "backwards_penalty" in self.attrs["reward"]:
                penalty = self.attrs["reward"]["backwards_penalty"]
                old_reward_func = self._get_reward
                def wrapper(agent, fish):
                    reward, truncated = old_reward_func(agent, fish)
                    if agent[3] < 0:  # Velocity less than 0
                        return reward - penalty, truncated
                    else:
                        return reward, truncated
                self._get_reward = wrapper

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

    def _parse_settings(self, name):
        settings = yaml.safe_load(open(FishEnv.fish_settings_yaml, "r"))
        attrs = {}
        attrs["from"] = name
        encountered_names = set()

        # Read in settings for dataset
        while "from" in attrs:
            from_name = attrs["from"]
            if from_name in encountered_names:
                raise ValueError("Recursive definition: {} depends on itself".format(name))
            encountered_names.add(name)
            old_attrs = attrs
            attrs = settings[from_name]

            # Overwrite attributes in the from dataset with the ones that are present in the top level dataset
            for key, value in old_attrs.items():
                if not key == "from":
                    if key in attrs:
                        # Helper function to merge properties that conflict
                        def helper(parent, key, child):
                            if type(parent[key]) is not dict or type(child) is not dict:
                                parent[key] = child
                            else:
                                existing_attrs = set(parent[key].keys())
                                new_attrs = set(child.keys())
                                conflict = existing_attrs.intersection(new_attrs)
                                for k in new_attrs.difference(conflict):
                                    parent[key][k] = child[k]
                                for k in conflict:
                                    helper(parent[key], k, child[k])
                        helper(attrs, key, value)
                    else:
                        attrs[key] = value

        return attrs

    def _get_obs(self):
        return np.vstack([self._agent, self._fish])

    def _get_info(self):
        # TODO: provide useful diagnostics here
        return {}

    def _action_to_vels(self, action):
        return np.array([self.max_v, self.max_omega]) * np.array([
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1]
        ])[action]
        # return np.array([self.max_v, self.max_omega]) * np.array([
        #     [-1, -1],
        #     [-1, 0],
        #     [-1, 1],
        #     [0, -1],
        #     [0, 0],
        #     [0, 1],
        #     [1, -1],
        #     [1, 0],
        #     [1, 1]
        # ])[action]

    def step(self, action):
        truncated = False
        if self.type == "sim":
            self._agent, self._fish = update(self.bounds, self.np_random, self._agent, self._fish, self._action_to_vels(action), self.dt, self.attrs["dynamics"])
        else:
            self._rl_controller.step()
            t, self._agent = self._rl_controller.get_robot_state()
            if self.attrs["robot_detection_timeout"] > 0 and time.time() - t > self.attrs["robot_detection_timeout"]:
                print("Episode truncated because robot was not detected")
                truncated = True
            _, self._fish = update(self.bounds, self.np_random, self._agent, self._fish, np.array([0, 0]), self.dt, self.attrs["dynamics"])
            self._rl_controller.command_vels(*self._action_to_vels(action))

        # Check if the agent is out of bounds
        if self._agent[0] < self.minx or self._agent[0] > self.maxx or self._agent[1] < self.miny or self._agent[1] > self.maxy:
            reward = -.25  # Assign a negative reward for touching the wall or going out of bounds
            terminated = False  # Terminate episode if agent is out of bounds
        else:
            reward, terminated = self._get_reward(self._agent, self._fish)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode in ["human", "camera", "threshold"]:
            self._render_frame()

        self._elapsed_steps += 1
        if self._max_episode_steps is not None and self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):

        if seed == None :
            seed = self.seed
    
        super().reset(seed=seed)

        # Initialize the states of the robot and fish
        self._agent, self._fish = init(self.bounds, self.n_fish, self.np_random, self.attrs["init_bounds"])
        if self.type == "real":
            # Wait for camera to detect robot initially
            self._rl_controller.reset()
            self._agent = None
            start_time = time.time()
            if self.attrs["robot_detection_reset_timeout"] < 0:
                self._rl_controller.init_state(np.zeros(5))
                _, self._agent = self._rl_controller.get_robot_state()
            while self._agent is None and (time.time() - start_time) < self.attrs["robot_detection_reset_timeout"]:
                _, self._agent = self._rl_controller.get_robot_state()
            if self._agent is None:
                raise ValueError("Timed out while trying to detect robot")
            self._rl_controller.command_vels(0, 0)

        self._elapsed_steps = 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode in ["human", "camera", "threshold"]:
            self._render_frame()

        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _scale_to_window(self, x):
        x = np.array(x)
        w = self.window_width - 2 * self.window_margin
        return w * x / self.bounds[1][0]

    def _scale_to_bounds(self, x):
        x = np.array(x)
        w = self.window_width - 2 * self.window_margin
        return self.bounds[1][0] * x / w

    def _to_window_coords(self, coords):
        m = self.window_margin * 2
        w = self.window_width - m
        h = self.window_height - m
        x, y = np.array(coords).T
        x = w * (x - self.bounds[0][0]) / self.bounds[1][0] + m / 2
        y = h * (y - self.bounds[0][1]) / self.bounds[1][1] + m / 2
        return np.stack([x, self.window_height - y]).T

    def _render_frame(self):
        if self.window is None and self.render_mode in ["human", "camera", "threshold"]:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode in ["human", "camera", "threshold"]:
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        if self.render_mode in ["camera", "threshold"]:
            frame = self._rl_controller.get_render_frame()
            img = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "BGR")
            img = pygame.transform.scale(img, self._scale_to_window(self.bounds[1] - self.bounds[0]))
            rect = img.get_rect().move(self.window_margin, self.window_margin)
            canvas.blit(img, rect)

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
        
        # if len(self._fish) >= 2:
        #     cov = np.cov(self._fish[:, :2].T)
        #     l1 = (cov[0][0] + cov[1][1]) / 2 + np.sqrt(np.square((cov[0][0] - cov[1][1]) / 2) + np.square(cov[0][1]))
        #     l2 = (cov[0][0] + cov[1][1]) / 2 - np.sqrt(np.square((cov[0][0] - cov[1][1]) / 2) + np.square(cov[0][1]))
        #     if cov[0][1] == 0:
        #         if cov[0][0] >= cov[1][1]:
        #             theta = 0
        #         else:
        #             theta = np.pi / 2
        #     else:
        #         theta = np.arctan2(l1 - cov[0][0], cov[0][1])
        #     x = np.average(self._fish[:, 0])
        #     y = np.average(self._fish[:, 1])
        #     w = 2 * np.sqrt(l1)  # 1st standard deviation
        #     h = 2 * np.sqrt(l2)
        #     target_rect = pygame.Rect([
        #         *self._to_window_coords([x-w/2, y+h/2]),
        #         *self._scale_to_window([w, h])
        #     ])
        #     shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        #     pygame.draw.ellipse(
        #         shape_surf,
        #         (255, 0, 0),
        #         [0, 0, *target_rect.size],
        #         3
        #     )
        #     pygame.draw.line(
        #         shape_surf,
        #         (255, 0, 0),
        #         [target_rect.size[0] / 2, 0],
        #         [target_rect.size[0] / 2, target_rect.size[1]],
        #         3
        #     )
        #     pygame.draw.line(
        #         shape_surf,
        #         (255, 0, 0),
        #         [0, target_rect.size[1] / 2],
        #         [target_rect.size[0], target_rect.size[1] / 2],
        #         3
        #     )
        #     rotated_surf = pygame.transform.rotate(shape_surf, theta * 180 / np.pi)
        #     canvas.blit(rotated_surf, rotated_surf.get_rect(center=target_rect.center))

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

        if self.render_mode in ["human", "camera", "threshold"]:
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
        if self.type == "real":
            self._rl_controller.close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Definition of FishEnv, the environment used to run both the simulated robot and the real robot"
    )
    parser.add_argument("-n", "--name", type=str, default="DefaultSim", help="name of the settings to use")
    parser.add_argument("-r", "--render_mode", type=str, default="human", help="how to render the environment")
    parser.add_argument("-s", "--seed", type=int, help="seed for the ennvironment")
    args = parser.parse_args()

    a = np.array([0, 0], dtype=np.int32)
    # a_map = {-1: {-1: 0, 0: 1, 1: 2}, 0: {-1: 3, 0: 4, 1: 5}, 1: {-1: 6, 0: 7, 1: 8}}
    a_map = {0: {-1: 0, 0: 1, 1: 2}, 1: {-1: 3, 0: 4, 1: 5}}
    env = FishEnv(args.name, seed=args.seed, render_mode=args.render_mode)

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
                # if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                #     a[0] = -1
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
    last_s = None
    dx = []
    dy = []
    # action_log = [2, 2, 2, 4, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 0, 1, 5, 5, 4, 5, 4, 5, 4, 5, 5, 5, 3, 5, 3, 3, 3, 2, 3, 2, 3, 2, 2, 3, 2, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 3, 3, 2, 2, 3, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 4, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 4, 5, 4, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 2, 2, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 5, 5, 2, 5, 5, 5, 4, 5, 4, 1, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 2, 5, 2, 2, 5, 5, 3, 3, 5, 3, 3, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 3, 2, 0, 2, 0, 2, 3, 0, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1, 2, 5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 5, 3, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 4, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 4, 5, 4, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 4, 4, 5, 4, 4, 0, 0, 0, 0, 5, 5, 5, 1, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 2, 5, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 0, 1, 2, 1, 3, 2, 1, 3, 3, 2, 3, 3, 2, 3, 2, 3, 2, 3, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 0, 0, 5, 5, 5, 2, 2, 5, 5, 5, 5, 4, 5, 4, 5, 4, 5, 2, 5, 5, 5, 5, 5, 3, 3, 5, 3, 3, 2, 3, 3, 2, 3, 2, 3, 2, 2, 1, 1, 5, 5, 5, 5, 4, 5, 4, 0, 2, 2, 5, 5, 5, 2, 2, 2, 2, 3, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 5, 3, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 5, 4, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 2, 5, 5, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 0, 0, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 3, 5, 5, 3, 3, 5, 2, 2, 3, 2, 2, 3, 2, 3, 2, 0, 2, 5, 5, 4, 5, 4, 5, 5, 4, 5, 5, 5, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 2, 2, 3, 2, 3, 2, 2, 1, 5, 1, 5, 4, 5, 4, 5, 0, 2, 2, 5, 5, 5, 2, 2, 2, 2, 3, 2, 3, 3, 3, 2, 3, 2, 2, 1, 1, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 3, 5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 3, 3, 3, 5, 3, 2, 3, 2, 2, 2, 2, 1, 1, 2, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 2, 3, 3, 2, 2, 3, 2, 2, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 1, 1, 5, 1, 5, 1, 5, 4, 1, 5, 1, 5, 1, 5, 0, 0, 0, 0, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 0, 0, 0, 0, 2, 5, 5, 5, 0, 0, 0, 0, 5, 5, 5, 5, 0, 4, 4, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 3, 3, 3, 3, 5, 3, 5, 5, 5, 3, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5]
    # # action_log = [2, 2, 2, 4, 2, 3, 3, 2, 3, 3, 3, 1]
    # # action_log = [0, 0, 0, 4, 0, 5, 5, 0, 5, 5, 5]
    # # action_log = []
    # action_history = []
    # state_history = []
    # i = 0
    while True:
        register_input()
        action = a_map[a[0]][a[1]]
        # action = action_log[i] if i < len(action_log) else a_map[a[0]][a[1]]
        # i += 1
        # if i >= len(action_log):
        #     quit = True
        # action_history.append(action)
        s, r, terminated, truncated, info = env.step(action)
        # state_history.append(s)
        if last_s is not None:
            dx.append(s[0, 0] - last_s[0, 0])
            dy.append(s[0, 1] - last_s[0, 1])
        last_s = s
        total_reward += r
        if steps % 200 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated or restart or quit:
            break
    env.close()
    # import matplotlib.pyplot as plt
    # plt.plot(dx)
    # plt.plot(dy)
    # plt.show()