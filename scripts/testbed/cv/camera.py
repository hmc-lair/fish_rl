
import numpy as np
import cv2
try:
    from .calibration import load_calibration_params, apply_calibration, undistort_and_warp
except:
    from calibration import load_calibration_params, apply_calibration, undistort_and_warp

class VideoProcessor:
    """Class to handle video processing, displaying, CV, etc."""

    def __init__(self, camera_port, calibration_path, render_mode=None, use_pygame=False):
        self._camera_port = camera_port
        self._calibration_params = load_calibration_params(calibration_path)
        self._bounds = self._calibration_params["CAMERA_BOUNDS"]
        self._height = self._bounds[1,1] - self._bounds[0,1]
        self._width = self._bounds[1,0] - self._bounds[0,0]
        self.render_mode = render_mode
        self.use_pygame = use_pygame

        self._cap = cv2.VideoCapture(self._camera_port)
        if self.render_mode in ["camera", "threshold"] and not self.use_pygame:
            cv2.namedWindow(self.render_mode, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.render_mode, 640, 480)

        self.reset()

    def reset(self):
        self.step()

    def step(self):
        ret, raw_frame = self._cap.read()
        if not ret:
            raise ValueError("Could not read from camera")
        self.raw_frame = raw_frame
        self.frame = apply_calibration(self.raw_frame, self._calibration_params)
        # self.frame = undistort_and_warp(self.raw_frame, self._calibration_params)
        self.render_frame = self.frame.copy()

    def get_coords(self, num_objects, offset=0):
        """Finds the n largest dark objects and returns their centroids in order"""
        coords = np.zeros([num_objects, 2])

        ## Orange thresholding for the robot to follow the orange dots
        lure_hsv =cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            #40, 45, 81
        # lower_blue = np.array([100,50,120], dtype = "uint8")  # 100, 50, 0
        # upper_blue = np.array([140,255,255], dtype = "uint8")  # 140, 255, 255

        lower_blue = np.array([0,0,0], dtype = "uint8")  # 100, 50, 0
        upper_blue = np.array([210,90,90], dtype = "uint8")  # 140, 255, 255

        lure_mask=cv2.inRange(lure_hsv,lower_blue,upper_blue)
        kernellure = np.ones((10,10),np.uint8)
        orange_closing = cv2.morphologyEx(lure_mask, cv2.MORPH_CLOSE, kernellure)
        orange_dilation = cv2.dilate(orange_closing, None, 1)
        
        if self.render_mode == "threshold":
            self.render_frame = cv2.cvtColor(255 - orange_dilation, cv2.COLOR_GRAY2RGB)

        cnts, hierarchy = cv2.findContours(orange_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # if len(cnts) < 2:
        #     print("cannot detect lure")
        cnts = cnts[0:2]

        if len(cnts) < num_objects: return False, coords  # if there aren't enough contours, return
        for i in range(0, num_objects):
            M = cv2.moments(cnts[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                met_cX, met_cY = self._pixels_to_meters([cX, cY], offset)

            else: cX, cY = 0, 0
            # cv2.circle(self.render_frame, (cX, cY), int(5/(i+1)), (320, 159, 22), -1)
            coords[i,:] = np.array([met_cX, met_cY])
        return True, coords
    
    def get_robot_state(self, robot_offset):
        valid, [head, tail] = self.get_coords(2, offset=robot_offset)
        fish_vect = head - tail
        theta = np.arctan2(fish_vect[1], fish_vect[0])
        robot_pos = (head + tail) / 2
        robot_state = (robot_pos[0], robot_pos[1], theta)
        if self.render_mode in ["camera", "threshold"]:
            cv2.circle(self.render_frame, self._meters_to_pixels(robot_pos, robot_offset), 3, (255, 0, 0), -1)
            if not self.use_pygame:
                cv2.imshow(self.render_mode, self.render_frame)
                cv2.waitKey(1)
        return valid, robot_state

    def get_fish_thresh(self):
        # fish_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # fish_kernel_open = np.ones((3,3),np.uint8)
        # fish_kernel_closed = np.ones((5,5),np.uint8)

        # fish_lower = np.array([0,35,0], dtype = "uint8")
        # fish_upper = np.array([25,255,230], dtype = "uint8")

        # colorfish = cv2.inRange(fish_hsv, fish_lower, fish_upper)

        # fish_mask_opening = cv2.morphologyEx(colorfish, cv2.MORPH_OPEN, fish_kernel_open)
        # fish_mask_opening2 = cv2.morphologyEx(fish_mask_opening, cv2.MORPH_OPEN, fish_kernel_open)
        # fish_mask_closing = cv2.morphologyEx(fish_mask_opening2, cv2.MORPH_CLOSE, fish_kernel_closed)

        # Convert the image to grayscale
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise and improve contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding to deal with shadows
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # Apply morphological operations to remove small noise and fill gaps
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the morphed binary image
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        min_contour_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Draw contours on the original image
        cv2.drawContours(self.render_frame, filtered_contours, -1, (0, 255, 0), 2)

        if self.render_mode in ["camera", "threshold"]:
            if not self.use_pygame:
                cv2.imshow(self.render_mode, self.render_frame)
                cv2.waitKey(1)

        # return fish_mask_closing

    def _pixels_to_meters(self, coords, offset=0):
        x, y = np.array(coords).T
        meters = np.array([  # convert from pixels to meters
            x,
            self._height - y
        ]) * self._calibration_params["METERS_PER_PIXEL"]

        # Adjust for offset from the chessboard
        depth = self._calibration_params["CAMERA_HEIGHT"]
        scale_factor = (depth - offset) / depth
        vec = np.array([self._width / 2, self._height / 2]) * self._calibration_params["METERS_PER_PIXEL"]
        return scale_factor * (meters - vec) + vec
    
    def _meters_to_pixels(self, coords, offset=0):
        scaled_meters = np.array(coords)

        # Adjust for offset from the chessboard
        depth = self._calibration_params["CAMERA_HEIGHT"]
        scale_factor = (depth - offset) / depth
        vec = np.array([self._width / 2, self._height / 2]) * self._calibration_params["METERS_PER_PIXEL"]
        meters = (scaled_meters - vec) / scale_factor + vec

        x, y = meters.T
        return np.array([  # convert from meters to pixels
            x / self._calibration_params["METERS_PER_PIXEL"],
            self._height - y / self._calibration_params["METERS_PER_PIXEL"]
        ]).astype(int)

    def close(self):
        cv2.destroyAllWindows()
        self._cap.release()
    
# if __name__ == '__main__':

#     camera_index = 0
#     camera_bounds = np.array([[687, 396], [1483, 801]]) # find these with calibrate_setup.py
#     vp = VideoProcessor(camera_index, camera_bounds, True)
#     while True:
#         vp.get_coords(2) #should this be 2?? 
#         #vp.display()
#         if not vp.is_go(): break
#     vp.close()

if __name__ == "__main__":
    video = VideoProcessor(2, "../config/calibration_keck.yaml", render_mode="camera")

    result = cv2.VideoWriter("../media/06_10_2024_16_38.avi",
                             cv2.VideoWriter_fourcc(*"MJPG"),
                             10,
                             (video.frame.shape[1], video.frame.shape[0])
                            )
    try:
        while True:
            video.step()
            result.write(video.frame)
            video.get_fish_thresh()
    except KeyboardInterrupt:
        print("Interrupted")
    
    result.release()
    video.close()
