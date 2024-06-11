import numpy as np
from testbed.cyberbot.state_machine import *
import time
from simple_pid import PID
import math
from testbed.cyberbot.rl_controller import RLController

class PIDState(State):
    def __init__(self, threshold, func):
        super().__init__()
        self._threshold = threshold
        self._func = func
    
    def run(self):
        super().run()
        if np.abs(self._func()) < self._threshold:
            self.end()

class Boolean(State):
    def __init__(self, bfunc):
        super().__init__()
        self._bfunc = bfunc
    
    def run(self):
        super().run()
        if not self._bfunc():
            self.end()

if __name__ == "__main__":
    # Setup serial communication to robot and the video processing code
    c = RLController("/dev/ttyACM0", 2, "../config/calibration.yaml", "../config/robot.yaml")

    # Trajectory for the robot to follow
    # points = np.array([
    #     [0.04, 0.26],
    #     [.1, .14],
    #     [0.32, 0.12],
    #     [.55, .13],
    #     [0.66, 0.22],
    #     [.58, .36],
    #     [0.38, 0.34],
    #     [.13, .35]
    # ])
    points = np.array([
        [0.40, 0.14],
        [0.58, 0.22],
        [0.40, 0.30],
        [0.13, 0.22]
    ])
    current_point_idx = 0
    current_robot_state = np.zeros(5)
    pid_lin_vel = PID(1.5, 0, 0, setpoint=0, output_limits=(-0.4147, 0.4147))  #KP, KI, KD, setpoint, limits
    pid_theta = PID(2, 0, 0, setpoint=0, output_limits=(-7.5812, 7.5812))  #Lin Vel: KP, KI, KD, setpoint, limits
    commanded_vels = np.array([0, 0], dtype=np.float64)  # v and omega

    def inc_point_idx():
        '''Move to next waypoint'''
        global current_point_idx
        current_point_idx += 1

    def control_theta():
        '''Use PID to control orientation to target point'''
        global commanded_vels
        x, y, theta, v, omega = current_robot_state
        desired_x, desired_y = points[current_point_idx]
        desired_theta = np.arctan2(desired_y - y, desired_x - x)
        control_effort_theta = pid_theta(RLController.wrap_to_pi(theta - desired_theta))
        commanded_vels += [0, control_effort_theta]
        print(f"Controlling theta to {desired_theta}")
        return control_effort_theta

    def control_lin_vel():
        '''Use PID to control distance to target point'''
        global commanded_vels
        x, y, theta, v, omega = current_robot_state
        desired_x, desired_y = points[current_point_idx]
        dist_x = x - desired_x
        dist_y = y - desired_y
        euclidian_dist = math.sqrt(dist_x**2 + dist_y**2)
        lin_vel_control_effort = pid_lin_vel(-euclidian_dist)
        commanded_vels += [lin_vel_control_effort, 0]
        print(f"Controlling distance {euclidian_dist}")
        return lin_vel_control_effort
    
    def face_zero():
        '''Use PID to face the robot left, i.e. to theta = 0'''
        global commanded_vels
        x, y, theta, v, omega = current_robot_state
        control_effort_theta = pid_theta(theta)
        commanded_vels += [0, control_effort_theta]
        print(f"Controlling theta to 0 with effort: {control_effort_theta}")
        return control_effort_theta

    # State machine that follows the waypoints once
    sm = Sequence([
        WaitForAny([
            Repeat(
                Sequence([
                    PIDState(0.2, control_theta),
                    WaitForAny([
                        PIDState(0.05, control_lin_vel),
                        PIDState(0, control_theta)
                    ]),
                    Lambda(inc_point_idx)
                ])
            ),
            Boolean(lambda: current_point_idx < len(points))
        ]),
        Lambda(lambda: c.send_commands(0, 0))
    ])

    # sm = Repeat(
    #     Sequence([
    #         WaitForAny([
    #             Repeat(Lambda(lambda: c.command_vels(0.8, 4 * np.pi))),
    #             Timer(3)
    #         ]),
    #         WaitForAny([
    #             Repeat(Lambda(lambda: c.command_vels(0.8, 4 * np.pi))),
    #             Timer(3)
    #         ])
    #     ])
    # )

    # sm = Sequence([
    #     Timer(2),
    #     PIDState(0, face_zero),
    #     Lambda(lambda: c.command_vels(0, 0))
    # ])

    # Run state machine
    sm.start()
    while sm.is_running():
        c.step()
        commanded_vels = np.array([0, 0], dtype=np.float64)
        _, current_robot_state = c.get_robot_state()
        x, y, theta, v, omega = current_robot_state
        print(f"Robot state: [{x:.4f}, {y:.4f}, {theta:.2f}, {v:.2f}, {omega:.2f}]")
        sm.run()
        # c.command_vels(*commanded_vels)
        time.sleep(0.005)  # Highest possible command frequency is 0.0032 (312.5 commands per second)