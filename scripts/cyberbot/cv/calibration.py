#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import yaml
import sys

def compute_perspective_warp(w, h, rvec, chessboard_bounds, chessboard_width_pixels, chessboard_height_pixels):
    '''
    Compute a perspective warp matrix using the calibration parameters so that the chessboard is aligned vertically and horizontally with the frame
    '''
    rotate_matrix = cv2.getRotationMatrix2D((w/2, h/2), np.linalg.norm(rvec) * 180 / np.pi, 1)
    x, y = rotate_matrix.dot(np.hstack([chessboard_bounds, np.ones((len(chessboard_bounds), 1))]).T).T[0]
    outpts = np.float32([
        [0, 0],
        [chessboard_width_pixels, 0],
        [chessboard_width_pixels, chessboard_height_pixels],
        [0, chessboard_height_pixels]
    ])
    outpts[:, 0] += x
    outpts[:, 1] += y
    return cv2.getPerspectiveTransform(chessboard_bounds.astype(np.float32), outpts)

def load_calibration_params(path):
    '''
    Load the calibration parameters from the given filepath and then cast them to the correct types
    '''
    with open(path, "r") as f:
        calibration_params = yaml.safe_load(f)
    for key in ["MTX", "DIST", "RVEC", "TVEC", "CHESSBOARD_BOUNDS", "CAMERA_BOUNDS", "PERSPECTIVE_WARP"]:
        calibration_params[key] = np.array(calibration_params[key])

    camera_bounds = calibration_params["CAMERA_BOUNDS"]
    minx, maxx = sorted(camera_bounds[:, 0])
    miny, maxy = sorted(camera_bounds[:, 1])
    calibration_params["CAMERA_BOUNDS"] = np.array([[minx, miny], [maxx, maxy]])

    return calibration_params

def undistort_and_warp(frame, calibration_params):
    '''
    Apply the undistort and warp operations to the whole frame
    '''
    h, w = frame.shape[:2]
    frame = cv2.undistort(frame, calibration_params['MTX'], calibration_params['DIST'])
    frame = cv2.warpPerspective(frame, calibration_params['PERSPECTIVE_WARP'], (w, h))
    return frame

def crop_frame(frame, calibration_params):
    '''
    Crop the undistorted and warped frame to the camera bounds
    '''
    camera_bounds = calibration_params["CAMERA_BOUNDS"]
    return frame[camera_bounds[0, 1]:camera_bounds[1, 1], camera_bounds[0, 0]:camera_bounds[1, 0]]

def draw_camera_bounds(frame, calibration_params):
    '''
    Draw the camera bounds on the undistorted and warped frame
    Alternative to cropping for the purposes of displaying the image
    '''
    return cv2.rectangle(frame, calibration_params['CAMERA_BOUNDS'][0], calibration_params['CAMERA_BOUNDS'][1], (255, 0, 0), 3)

def apply_calibration(frame, calibration_params):
    '''
    Prepares a frame for use by the rest of the pipeline that detects objects
    Undistorts and warps the frame, then crops it to the camera bounds
    '''
    return crop_frame(undistort_and_warp(frame, calibration_params), calibration_params)

def calibrate(port_number, chessboard_size=None, chessboard_square_size_mm=None):
    '''
    Reads the camera feed for the given port number
    Attempts to reach the given fps
    If render is false, the camera feed is not displayed
    Returns the measured fps
    '''
    # Create display
    name = f"Camera feed from /dev/video{port_number}"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(name, 640 * 2, 480 * 2)

    # Code to record clicks
    camera_bounds = np.zeros((2, 2), dtype=int)
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            camera_bounds[0] = [x, y]
        if event == cv2.EVENT_RBUTTONDOWN:
            camera_bounds[1] = [x, y]
    cv2.setMouseCallback(name, on_mouse)
    
    # Attempt to start video capture
    cap = cv2.VideoCapture(port_number)
    if cap is None or not cap.isOpened():
       print(f'Unable to open video source: /dev/video{port_number}', file=sys.stderr)
       return 0

    # Try to get the first frame
    if cap.isOpened():
        rval, frame = cap.read()
        dst = frame.copy()
    else:
        rval = False

    # If a chessboard is detected, pressing space schedules callibration on the next `n_calibration_frames` frames
    n_calibration_frames = 10
    calibration_frames = -1 

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    mtx = dist = rvec = tvec = chessboard_bounds = chessboard_square_size_pixels = perspective_warp = None
    flags = cv2.CALIB_CB_FAST_CHECK

    # Detect keyboard interrupt
    try:
        # Process frames
        while rval:
            # Calibrate camera with the chessboard corners found
            if calibration_frames >= 0:
                if calibration_frames > 0:
                    if chessboard_found:
                        objpoints.append(objp)
                        imgpoints.append(corners)
                else:
                    if len(objpoints) != n_calibration_frames:
                        print(f"error: no chessboard detected in {n_calibration_frames - len(objpoints)} out of {n_calibration_frames} calibration frames", file=sys.stderr)
                    else:
                        # Calculate calibration parameters
                        h, w = frame.shape[:2]
                        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
                        rvec = np.average(rvecs, axis=0)
                        tvec = np.average(tvecs, axis=0)

                        # Compute the re-projection error. Closer to 0 is better
                        mean_error = 0
                        for imgp, objp, rvec, tvec in zip(imgpoints, objpoints, rvecs, tvecs):
                            reprojected_imgp, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
                            error = cv2.norm(imgp, reprojected_imgp, cv2.NORM_L2)
                            mean_error += error / len(objpoints)
                        print("re-projection error: {}".format(error), file=sys.stderr)

                        # Re-find chessboard corners after undistorting
                        dst2 = cv2.undistort(frame, mtx, dist)
                        _, undistorted_corners = cv2.findChessboardCorners(dst2, chessboard_size, flags=flags)

                        # Compute the bounds on the chessboard and the pixel size of a chessboard square
                        chessboard_bounds = undistorted_corners.reshape(-1, 2)[[0, chessboard_size[0]-1, -1, -chessboard_size[0]]]
                        chessboard_width_pixels = np.linalg.norm(chessboard_bounds[1] - chessboard_bounds[0]) / (chessboard_size[0] - 1)
                        chessboard_height_pixels = np.linalg.norm(chessboard_bounds[3] - chessboard_bounds[0]) / (chessboard_size[1] - 1)
                        chessboard_square_size_pixels = np.average([chessboard_width_pixels, chessboard_height_pixels])

                        # Compute a perspective warp so that the chessboard is aligned vertically and horizontally
                        perspective_warp = compute_perspective_warp(frame.shape[1], frame.shape[0], rvec, chessboard_bounds, chessboard_width_pixels, chessboard_height_pixels)
                calibration_frames -= 1

            # Rendering
            if mtx is not None and dist is not None and perspective_warp is not None:
                dst = cv2.undistort(dst, mtx, dist)
                dst = cv2.warpPerspective(dst, perspective_warp, (frame.shape[1], frame.shape[0]))
            dst = cv2.rectangle(dst, camera_bounds[0], camera_bounds[1], (255, 0, 0), 3)
            cv2.imshow(name, dst)

            # Read a new frame
            rval, frame = cap.read()
            dst = frame.copy()

            # Try to detect a chessboard
            chessboard_found, corners = cv2.findChessboardCorners(frame, chessboard_size, flags=flags)
            cv2.drawChessboardCorners(dst, chessboard_size, corners, chessboard_found)

            # Keypress handling
            key = cv2.waitKey(1)
            if key == 27:  # Escape key
                break
            elif key == 32:  # Space key
                if calibration_frames < 0:
                    calibration_frames = n_calibration_frames
                    objpoints = []  # 3d points in real world space
                    imgpoints = []  # 2d points in image plane
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
    
    # Cleanup
    cv2.destroyWindow(name)
    cap.release()

    # Return the calibration parameters in a dictionary
    if mtx is not None:
        return {
            "MTX": mtx.tolist(),
            "DIST": dist.tolist(),
            "RVEC": rvec.tolist(),
            "TVEC": tvec.tolist(),
            "PERSPECTIVE_WARP": perspective_warp.tolist(),
            "CHESSBOARD_SIZE": list(chessboard_size),
            "CHESSBOARD_BOUNDS": chessboard_bounds.tolist(),
            "CHESSBOARD_SQUARE_SIZE_PIXELS": chessboard_square_size_pixels.tolist(),
            "CHESSBOARD_SQUARE_SIZE_MM": chessboard_square_size_mm,
            "METERS_PER_PIXEL": (chessboard_square_size_mm / 1000 / chessboard_square_size_pixels).tolist(),
            "CAMERA_BOUNDS": camera_bounds.tolist()
        }
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute calibration parameters using a chessboard. Press space to calibrate once the chessboard is detected. Then click to set bounds. Press esc to finish.",
        epilog="To find which cameras are available, run `ls /dev | grep video` or `v4l2-ctl --list-devices`"
    )
    parser.add_argument("port_number", type=int, help="camera port number")
    parser.add_argument("corners_per_row", type=int, help="chessboard number of inner corners per row")
    parser.add_argument("corners_per_col", type=int, help="chessboard number of inner corners per column")
    parser.add_argument("square_size_mm", type=float, help="chessboard square size in mm")
    parser.add_argument("-o", "--output", help="file to write calibration parameters to")
    args = parser.parse_args()

    # Perform the calibration
    calibration_params = calibrate(
        args.port_number,
        chessboard_size=(args.corners_per_row, args.corners_per_col),
        chessboard_square_size_mm=args.square_size_mm
    )

    # Output callibration parameters to stdout or the provided file
    if calibration_params is not None:
        with open(args.output, "w+") if args.output is not None else sys.stdout as f:
            yaml.dump(calibration_params, f, default_flow_style=None)
