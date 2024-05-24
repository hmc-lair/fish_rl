#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse
import yaml

CHECKERBOARD_SQUARE_SIZE = 57.45

def display(port_number, fps=20, render=True, undistort=False, chessboard=None, chessboard_square_size_mm=None, output=None):
    '''
    Reads the camera feed for the given port number
    Attempts to reach the given fps
    If render is false, the camera feed is not displayed
    Returns the measured fps
    '''
    name = f"Camera feed from /dev/video{port_number}"

    camera_bounds = np.zeros((2, 2), dtype=int)
    num_clicks = 0
    def on_mouse(event, x, y, flags, param):
        nonlocal num_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            camera_bounds[num_clicks % 2] = [x, y]
            if num_clicks % 2 == 1:
                print("CAMERA_BOUNDS\n", camera_bounds)
            num_clicks += 1

    # Create window
    if render:
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, on_mouse)
    
    # Attempt to start video capture
    cap = cv2.VideoCapture(port_number)
    if cap is None or not cap.isOpened():
       print(f'Unable to open video source: /dev/video{port_number}')
       return 0
    
    # Get the camera's FPS
    print(f"CAP_PROP_FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    # Try to get the first frame
    if cap.isOpened():
        rval, frame = cap.read()
    else:
        rval = False
    
    # Timing information
    avg_delay = 0
    curr_time = time.time()
    n = 0
    if fps == 0:
        target_delay = 0
        print(f"Target fps is infinity")
        print(f"Target delay is 0 ms")
    else:
        target_delay = 1/fps
        print(f"Target fps is {fps}")
        print(f"Target delay is {int(1000 * target_delay)} ms")

    # Setting up calibration
    n_calibration_frames = 10
    calibration_frames = -1  # If a chessboard is detected, pressing space schedules callibration on the next n frames
    if chessboard is not None:
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1,2) * CHECKERBOARD_SQUARE_SIZE
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
    mtx = dist = rvec = tvec = None
    chessboard_bounds = chessboard_square_size_pixels = None

    # Detect keyboard interrupt
    try:
        # Read frames from camera
        while rval:
            # Calibrate camera with the chessboard corners found
            if calibration_frames >= 0:
                if calibration_frames > 0:
                    if retval:
                        objpoints.append(objp)
                        imgpoints.append(corners)
                else:
                    # Calculate calibration parameters
                    h, w = frame.shape[:2]
                    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
                    rvec = np.average(rvecs, axis=0)
                    tvec = np.average(tvecs, axis=0)
                    print(f"completed on {len(objpoints)}/{n_calibration_frames} frames")
                    print("MTX\n", mtx)
                    print("DIST\n", dist)
                    print("RVEC\n", rvec)
                    print("TVEC\n", tvec)

                    # Compute the bounds on the chessboard and the pixel size of a chessboard square
                    chessboard_bounds = np.zeros((len(imgpoints), 4, 2))
                    for i, imgp in enumerate(imgpoints):
                        chessboard_bounds[i] = corners.reshape(-1, 2)[[0, chessboard[0]-1, -1, -chessboard[0]]]
                    chessboard_bounds = np.average(chessboard_bounds, axis=0)
                    chessboard_width_pixels = np.linalg.norm(chessboard_bounds[1] - chessboard_bounds[0]) / chessboard[0]
                    chessboard_height_pixels = np.linalg.norm(chessboard_bounds[3] - chessboard_bounds[0]) / chessboard[1]
                    chessboard_square_size_pixels = np.average([chessboard_width_pixels, chessboard_height_pixels])
                    print("CHESSBOARD_BOUNDS\n", chessboard_bounds)
                    print("CAMERA_SQUARE_SIZE_PIXELS\n", chessboard_square_size_pixels)

                    # Compute the re-projection error. Closer to 0 is better
                    mean_error = 0
                    for imgp, objp, rvec, tvec in zip(imgpoints, objpoints, rvecs, tvecs):
                        reprojected_imgp, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
                        error = cv2.norm(imgp, reprojected_imgp, cv2.NORM_L2)
                        mean_error += error / len(objpoints)
                    print("re-projection error: {}".format(error))
                calibration_frames -= 1

            if render:
                dst = frame
                if mtx is not None:
                    dst = cv2.undistort(frame, mtx, dist)
                    h, w = frame.shape[:2]
                    rotate_matrix = cv2.getRotationMatrix2D((w/2, h/2), np.linalg.norm(rvec) * 180 / np.pi, 1)
                    x, y = rotate_matrix.dot(np.hstack([chessboard_bounds, np.ones((len(chessboard_bounds), 1))]).T).T[0]
                    outpts = np.float32([
                        [x, y],
                        [x + chessboard[0] * chessboard_square_size_pixels, y],
                        [x + chessboard[0] * chessboard_square_size_pixels, y + chessboard[1] * chessboard_square_size_pixels],
                        [x, y + chessboard[1] * chessboard_square_size_pixels]
                    ])
                    M = cv2.getPerspectiveTransform(chessboard_bounds.astype(np.float32), outpts)
                    dst = cv2.warpPerspective(dst, M, (w, h))
                    # dst = cv2.warpAffine(dst, rotate_matrix, dsize=(w, h))
                dst = cv2.rectangle(dst, camera_bounds[0], camera_bounds[1], (255, 0, 0), 3)
                cv2.imshow(name, dst)

            # Read a new frame
            rval, frame = cap.read()

            # Try to detect a chessboard
            if chessboard is not None:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH & \
                        cv2.CALIB_CB_NORMALIZE_IMAGE & \
                        cv2.CALIB_CB_FILTER_QUADS & \
                        cv2.CALIB_CB_FAST_CHECK
                retval, corners = cv2.findChessboardCorners(frame, chessboard, flags=flags)
                cv2.drawChessboardCorners(frame, chessboard, corners, retval)

            # Wait the given number of milliseconds
            # Break on escape key press
            remaining_delay = max(int(1000 * (curr_time + target_delay - time.time())), 1)
            key = cv2.waitKey(remaining_delay)
            if key == 27:
                break
            elif key == 32:
                if chessboard is not None and calibration_frames < 0:
                    calibration_frames = n_calibration_frames
                    objpoints = []  # 3d points in real world space
                    imgpoints = []  # 2d points in image plane
                    print("Calibrating camera... ", end="")
                elif chessboard is None:
                    print("Must provide chessboard size in order to calibrate camera")
                else:
                    print("No chessboard detected for calibration")

            # Update timing information
            n += 1
            new_time = time.time()
            diff = new_time - curr_time
            avg_delay = avg_delay * (n - 1) / n + diff / n
            curr_time = new_time
    except KeyboardInterrupt:
        print("Interrupted")
    
    # Cleanup
    if render:
        cv2.destroyWindow(name)
    cap.release()

    # Output callibration parameters
    if output is not None:
        if mtx is not None:
            print(f"Writing calibration parameters to {output}")
            calibration_params = {
                "MTX": mtx.tolist(),
                "DIST": dist.tolist(),
                "RVEC": rvec.tolist(),
                "TVEC": tvec.tolist(),
                "CHESSBOARD_BOUNDS": chessboard_bounds.tolist(),
                "CHESSBOARD_SQUARE_SIZE_PIXELS": chessboard_square_size_pixels.tolist(),
                "CHESSBOARD_SQUARE_SIZE_MM": chessboard_square_size_mm,
                "CAMERA_BOUNDS": camera_bounds.tolist()
            }
            with open(output, "w+") as f:
                yaml.dump(calibration_params, f, default_flow_style=None)
        else:
            print("No calibration parameters to write")

    # Return measured fps
    print(f"Avg delay was {avg_delay * 1000:.2f} ms")
    if avg_delay == 0:
        return 0
    else:
        return 1 / avg_delay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display a camera feed using OpenCV. Press esc to quit",
        epilog="To find which cameras are available, run `ls /dev | grep video` or `v4l2-ctl --list-devices`"
    )
    parser.add_argument("port_number", type=int)
    parser.add_argument("-f", "--fps", default=0, type=int, help="target FPS. Enter 0 to remove the scheduled delay between frames entirely")
    parser.add_argument("-l", "--headless", action="store_true", help="run without rendering the camera feed")
    parser.add_argument("-u", "--undistort", action="store_true", help="undistort fisheye")
    parser.add_argument("-c", "--chessboard", nargs="*", type=int, help="look for a chessboard. Takes 3 values: number of inner corners per row, number of inner corners per column, and chessboard square size in mm")
    parser.add_argument("-o", "--output", help="file to write calibration parameters out to")
    args = parser.parse_args()
    if args.chessboard is not None and len(args.chessboard) != 3:
        parser.error("argument -c/--chessboard: requires exactly 3 arguments")
    fps = display(args.port_number, fps=args.fps, render=not args.headless, undistort=args.undistort, chessboard=args.chessboard[:2], chessboard_square_size_mm=args.chessboard[2], output=args.output)
    print(f"Measured fps was {fps:.2f}")
