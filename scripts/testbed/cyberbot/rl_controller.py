import serial
import numpy as np
import time
from ..cv import camera
import yaml

class RLController():
    def wrap_to_pi(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def __init__(self, transmit_port, camera_port, calibration_path, robot_path, render_mode=None, use_pygame=False):
        self._transmit_port = transmit_port
        self._ser = serial.Serial(self._transmit_port, baudrate=115200)

        self._video = camera.VideoProcessor(camera_port, calibration_path, render_mode=render_mode, use_pygame=use_pygame)
        with open(robot_path, "r") as f:
            self._params = yaml.safe_load(f)
        self._wheel_separation = self._params["WHEEL_SEPARATION"] / 1000
        self._wheel_radius = self._params["WHEEL_RADIUS"] / 1000
        self._width = self._params["WIDTH"] / 1000
        self._length = self._params["LENGTH"] / 1000

        self._fishtank_bounds = (self._video._bounds - self._video._bounds[0]) * self._video._calibration_params["METERS_PER_PIXEL"]
        margin = max(self._width, self._length)
        self._reachable_range = np.stack([self._fishtank_bounds[0] + margin, self._fishtank_bounds[1] - margin])

        self.reset()

    def reset(self):
        self._timestamp = None
        self._state = None
        self._video.reset()

    def _get_wheel_vels(self, v, omega):
        """Convert linear and angular velocity command to wheel velocities"""
        if omega == 0:
            vR = vL = v
        else:
            vR = omega * (v / omega + self._wheel_separation / 2)
            vL = omega * (v / omega - self._wheel_separation / 2)
        
        # Convert to radians per second
        vR /= self._wheel_radius
        vL /= self._wheel_radius

        # Convert to 1/64th wheel increments per second
        vR *= 32 / np.pi
        vL *= 32 / np.pi

        # Preserve ratio between wheel velocities when capping to 128
        if vR != 0 or vL != 0:
            signs = np.sign([vR, vL])
            vels = np.abs([vR, vL])
            max_idx = np.argmax(vels)
            min_idx = np.argmin(vels)
            ratio = vels[min_idx] / vels[max_idx]
            vels = np.minimum(vels, [128, 128])
            vels[min_idx] = vels[max_idx] * ratio
            vels *= signs

        return vR, vL

    def send_commands(self, vR, vL):
       """Sends commands to the transmit board over the serial port"""
       #print("vL = " + str(vL) + ", vR = " + str(vR))
       dict = {'vL': vL, 'vR' : vR}
       packet = str(dict) + "\r"
       self._ser.write(packet.encode())

    def command_vels(self, v, omega):
        """Send a linear and angular velocity command to the robot"""

        #cap linear velocity
        heading = self._state[2]
        v_vec = np.array([np.cos(heading),  np.sin(heading)]) * v

        if self._state[0] <= self._reachable_range[0][0]:
            print("Left wall")
            if v_vec @ np.array([1,0]) < 0: #facing into wall on left
                v = -self._state[3]
        
        if self._state[0] >= self._reachable_range[1][0]:
            print("Right Wall")
            if v_vec @ np.array([-1, 0]) < 0: #facing right wall
                v = -self._state[3]
        
        if self._state[1] <= self._reachable_range[0][1]:
            print("Bottom Wall")
            if v_vec @ np.array([0, 1]) < 0: #facing into wall on left
                v = -self._state[3]
        
        if self._state[1] >= self._reachable_range[1][1]:
            print("Top Wall")
            if v_vec @ np.array([0, -1]) < 0: #facing into wall on left
                v = -self._state[3]

        self.send_commands(*self._get_wheel_vels(v, omega))
    
    def get_robot_state(self):
        '''
        Returns the robot state and the timestamp at which it was taken
        '''
        valid, proposed_state = self._video.get_robot_state(self._params["OFFSET"])
        if valid:
            timestamp = time.time()
            if self._timestamp is None:
                self._state = np.array([*proposed_state, 0, 0])
            else:
                dt = timestamp - self._timestamp
                prev_x, prev_y, prev_theta, prev_v, prev_omega = self._state
                x, y, theta = proposed_state
                meas_v = np.linalg.norm([x - prev_x, y - prev_y]) / dt * np.sign((x - prev_x) * np.cos(theta) + (y - prev_y) * np.sin(theta))
                meas_omega = RLController.wrap_to_pi(theta - prev_theta) / dt

                # Apply a first-order IIR filter (infinite impulse response)
                # Essentially acts as a low pass filter with a time constant dependent on
                # the sampling period and the coefficient used (ff):
                # http://www.tsdconseil.fr/tutos/tuto-iir1-en.pdf
                v_ff = 0.4
                omega_ff = 0.25
                v = (1 - v_ff) * prev_v + v_ff * meas_v
                omega = (1 - omega_ff) * prev_omega + omega_ff * meas_omega

                self._state = np.array([x, y, theta, v, omega])
            self._timestamp = timestamp
        return self._timestamp, self._state

    def get_render_frame(self):
        return self._video.render_frame

    def close(self):
        self.command_vels(0, 0)
        self._ser.close()
        self._video.close()

