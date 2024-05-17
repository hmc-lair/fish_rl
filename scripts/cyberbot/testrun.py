import serial
import numpy as np
from StateMachine import *
import time
from cv import camera 
from simple_pid import PID
import math

class RLController():
    def __init__(self, camera_port=0, camera_bounds = np.array([[636,  302], [1500, 782]]), transmit_port='/dev/tty.usbmodem14102'):
        self._ser = serial.Serial(transmit_port, baudrate=115200)
        self._video = camera.VideoProcessor(camera_port, camera_bounds)

    def send_commands(self, vR, vL):
       """Sends commands to the transmit board over the serial port"""
       #print("vL = " + str(vL) + ", vR = " + str(vR))
       dict = {'vL': vL, 'vR' : vR}
       packet = str(dict) + "\r"
       print(packet)
       self._ser.write(packet.encode())

    def get_video_state(self):
        proposed_state = self._video.get_robot_state()
        print(proposed_state)
        return proposed_state

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

def wrap_to_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

if __name__ == '__main__':
    # Setup serial communication to robot and the video processing code
    camera_bounds = np.array([[636,  302], [1500, 782]])  # find these with calibrate_setup.py
    port_t = '/dev/ttyACM0'                     # find this with ls /dev | grep usbmodem
    port_c = 0                                            # either 0 or 1
    c = RLController(camera_bounds=camera_bounds, camera_port=port_c, transmit_port=port_t)

    # Trajectory for the robot to follow
    points = np.array([
        [0.04, 0.26],
        [.1, .14],
        [0.32, 0.12],
        [.55, .13],
        [0.66, 0.22],
        [.58, .36],
        [0.38, 0.34],
        [.13, .35]
    ])
    current_point_idx = 0
    current_robot_state = (0, 0, 0)
    pid_theta = PID(30, 0, 0, setpoint=0, output_limits=(-100, 100))  #KP, KI, KD, setpoint, limits
    pid_lin_vel= PID(200, 0, 0, setpoint=0, output_limits=(-100, 100))  #Lin Vel: KP, KI, KD, setpoint, limits
    commanded_vels = np.array([0, 0], dtype=np.float64)  # vR and vL

    def inc_point_idx():
        '''Move to next waypoint'''
        global current_point_idx
        current_point_idx += 1

    def control_theta():
        '''Use PID to control orientation to target point'''
        global commanded_vels
        x, y, theta = current_robot_state
        desired_x, desired_y = points[current_point_idx]
        desired_theta = np.arctan2(desired_y - y, desired_x - x)
        control_effort_theta = pid_theta(wrap_to_pi(theta - desired_theta))
        commanded_vels += np.array([control_effort_theta, -control_effort_theta])
        print(f"Controlling theta to {desired_theta}")
        return control_effort_theta

    def control_lin_vel():
        '''Use PID to control distance to target point'''
        global commanded_vels
        x, y, theta = current_robot_state
        desired_x, desired_y = points[current_point_idx]
        dist_x = x - desired_x
        dist_y = y - desired_y
        euclidian_dist = math.sqrt(dist_x**2 + dist_y**2)
        lin_vel_control_effort = pid_lin_vel(-euclidian_dist)
        commanded_vels += np.array([lin_vel_control_effort, lin_vel_control_effort])
        print(f"Controlling distance {euclidian_dist}")
        return lin_vel_control_effort
    
    def face_left():
        '''Use PID to face the robot left, i.e. to theta = 0'''
        global commanded_vels
        control_effort_theta = pid_theta(theta)
        commanded_vels += np.array([control_effort_theta, -control_effort_theta])
        print(f"Controlling theta to 0")
        return control_effort_theta

    # State machine that follows the waypoints once
    sm = Sequence([
        WaitForAny([
            Repeat(
                Sequence([
                    PIDState(10, control_theta),
                    WaitForAny([
                        PIDState(10, control_lin_vel),
                        PIDState(0, control_theta)
                    ]),
                    Lambda(inc_point_idx)
                ])
            ),
            Boolean(lambda: current_point_idx < len(points))
        ]),
        Lambda(lambda: c.send_commands(0, 0))
    ])

    # Run state machine
    sm.start()
    while sm.is_running():
        commanded_vels = np.array([0, 0], dtype=np.float64)
        current_robot_state = c.get_video_state()
        sm.run()
        c.send_commands(*commanded_vels)
        time.sleep(0.005)  # Highest possible command frequency is 0.0032 (312.5 commands per second)