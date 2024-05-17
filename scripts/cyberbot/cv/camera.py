
import numpy as np
import cv2
import os
from datetime import datetime
from cv import convert

PIX2METERS = .635/820 # meters/pixels conversion TODO: automate this calculation in __init__
FPS = 10

# MTX and DIST are properties of the camera (have to do with fisheye lens)
MTX = np.array([[1.05663779e+03, 0.00000000e+00, 9.73055094e+02],
 [0.00000000e+00, 1.05269643e+03, 5.64799418e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DIST = np.array([-3.80359934e-01,  1.49531854e-01,  2.50649988e-05,  8.39488578e-05,  -2.83529982e-02])

class VideoProcessor:
    """Class to handle video processing, displaying, CV, etc."""

    def __init__(self, camera_port, camera_bounds):
        self._cap = cv2.VideoCapture(camera_port)
        self._height = camera_bounds[1,1]-camera_bounds[0,1]
        self._current_frame = None
        self._bounds = camera_bounds

    def get_coords(self, num_objects):
        """Finds the n largest dark objects and returns their centroids in order"""

        coords = np.zeros([num_objects, 2])
        ret, frame = self._cap.read()

        frame = cv2.undistort(frame, MTX, DIST, None, MTX)
        frame = frame[self._bounds[0][1]:self._bounds[1][1], self._bounds[0][0]:self._bounds[1][0]]
        self._current_frame = frame.copy()

        if (ret is None or frame is None): return coords # if frame isn't valid, return

        ## Orange threshhlding for the robot to follow the orange dots

        lure_hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
 
        lower_blue = np.array([100,170,50], dtype = "uint8") #100, 50, 0
        upper_blue = np.array([120,255,255], dtype = "uint8") #140, 255, 255

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

                met_cX = convert.xpxtomet(cX)
                met_cY = convert.ypxtomet(self._height - cY)

            else: cX, cY = 0, 0
            cv2.circle(self._current_frame, (cX, cY), int(5/(i+1)), (320, 159, 22), -1)
            coords[i,:] = np.array([met_cX, met_cY])
        # coords[:,1] = (convert.ypxtomet(self._height)) - coords[:,1] # move origin to lower left corner
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
            xpx = convert.xmettopx(target.x)
            ypx = convert.ymettopx(target.y)
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