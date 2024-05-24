
import numpy as np
import cv2
from cv.calibration import load_calibration_params, apply_calibration

class VideoProcessor:
    """Class to handle video processing, displaying, CV, etc."""

    def __init__(self, camera_port, calibration_path):
        self._calibration_params = load_calibration_params(calibration_path)
        self._cap = cv2.VideoCapture(camera_port)
        self._bounds = self._calibration_params["CAMERA_BOUNDS"]
        self._height = self._bounds[1,1] - self._bounds[0,1]
        self._current_frame = None

    def get_coords(self, num_objects):
        """Finds the n largest dark objects and returns their centroids in order"""

        coords = np.zeros([num_objects, 2])
        ret, frame = self._cap.read()

        frame = apply_calibration(frame, self._calibration_params)
        self._current_frame = frame.copy()

        if (ret is None or frame is None): return coords # if frame isn't valid, return

        ## Orange thresholding for the robot to follow the orange dots

        lure_hsv =cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
        lower_blue = np.array([100,50,120], dtype = "uint8")  # 100, 50, 0
        upper_blue = np.array([140,255,255], dtype = "uint8")  # 140, 255, 255

        lure_mask=cv2.inRange(lure_hsv,lower_blue,upper_blue)
        kernellure = np.ones((10,10),np.uint8)
        orange_closing = cv2.morphologyEx(lure_mask, cv2.MORPH_CLOSE, kernellure)
        orange_dilation = cv2.dilate(orange_closing, None, 1)
        
        cv2.imshow("orange thresh",orange_dilation)
        cv2.waitKey(1)

        cnts, hierarchy = cv2.findContours(orange_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(cnts) <2:
            print("cannot detect lure")
        cnts = cnts[0:2]

        if len(cnts) < num_objects: return coords # if there aren't enough contours, return
        for i in range(0, num_objects):
            M = cv2.moments(cnts[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                met_cX = cX * self._calibration_params['METERS_PER_PIXEL']
                met_cY = (self._height - cY) * self._calibration_params['METERS_PER_PIXEL']

            else: cX, cY = 0, 0
            cv2.circle(self._current_frame, (cX, cY), int(5/(i+1)), (320, 159, 22), -1)
            coords[i,:] = np.array([met_cX, met_cY])
        return coords
    
    def get_robot_state(self):
        [head, tail] = self.get_coords(2)
        fish_vect = head - tail
        theta = np.arctan2(fish_vect[1], fish_vect[0])
        robot_pos = (head + tail)/2
        robot_state = (robot_pos[0], robot_pos[1], theta)
        return robot_state
    
    def display(self, target):
        """Shows live video feed, plotting dots on identified objects and the bot target"""

        if self._current_frame is not None:
            xpx = target.x / self._calibration_params["METERS_PER_PIXEL"]
            ypx = target.y / self._calibration_params["METERS_PER_PIXEL"]
            cv2.circle(self._current_frame, (int(xpx), int(self._height-ypx)), 5, (0, 159, 22), -1)
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            resized = cv2.resize(self._current_frame, (960, 540))
            cv2.imshow('frame', resized)

    def cleanup(self):
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    camera_index = 0
    camera_bounds = np.array([[687, 396], [1483, 801]]) # find these with calibrate_setup.py
    vp = VideoProcessor(camera_index, camera_bounds, True)
    while True:
        vp.get_coords(2) #should this be 2?? 
        #vp.display()
        if not vp.is_go(): break
    vp.cleanup()