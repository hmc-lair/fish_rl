import serial
import numpy as np
import time
from ..cv import camera
import yaml

class RLController():
    def wrap_to_pi(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def __init__(self, transmit_port, camera_port, calibration_path, robot_path):
        self._ser = serial.Serial(transmit_port, baudrate=115200)
        self._video = camera.VideoProcessor(camera_port, calibration_path)
        with open(robot_path, "r") as f:
            self._params = yaml.safe_load(f)
        self._wheel_separation = self._params["WHEEL_SEPARATION"] / 1000
        self._wheel_radius = self._params["WHEEL_RADIUS"] / 1000
        self._timestamp = None
        self._state = None

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
        self.send_commands(*self._get_wheel_vels(v, omega))

    def get_video_state(self):
        proposed_state = self._video.get_robot_state(self._params["OFFSET"])
        # print(proposed_state)
        return proposed_state
    
    def get_robot_state(self):
        proposed_state = self._video.get_robot_state(self._params["OFFSET"])
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
        return self._state

