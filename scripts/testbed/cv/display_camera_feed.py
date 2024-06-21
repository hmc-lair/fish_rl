#!/usr/bin/env python3
import cv2
import time
import argparse
from calibration import load_calibration_params, apply_calibration
import numpy as np

def display(port_number, fps=20, render=True, calibration_params=None):
    '''
    Reads the camera feed for the given port number
    Attempts to reach the given fps
    If render is false, the camera feed is not displayed
    Returns the measured fps
    '''
    name = f"Camera feed from /dev/video{port_number}"

    # Create window
    if render:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 640, 480)

        if calibration_params is not None:
            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(f"Pixels: [{x}, {y}]. Meters: [{x * calibration_params['METERS_PER_PIXEL']:.4f}, {y * calibration_params['METERS_PER_PIXEL']:.4f}]")
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

    # Detect keyboard interrupt
    try:
        # Read frames from camera
        while rval:
            if calibration_params is not None:
                frame = apply_calibration(frame, calibration_params)
            if render:
                cv2.imshow(name, frame)

            # Read a new frame
            rval, frame = cap.read()

            # Wait the given number of milliseconds
            # Break on escape key press
            remaining_delay = max(int(1000 * (curr_time + target_delay - time.time())), 1)
            key = cv2.waitKey(remaining_delay)
            if key == 27:
                break

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

    # Return measured fps
    print(f"Avg delay was {avg_delay * 1000:.2f} ms")
    if avg_delay == 0:
        return 0
    else:
        return 1 / avg_delay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display a camera feed. Press esc to quit",
        epilog="To find which cameras are available, run `ls /dev | grep video` or `v4l2-ctl --list-devices`"
    )
    parser.add_argument("port_number")
    parser.add_argument("-f", "--fps", default=0, type=int, help="target FPS. Enter 0 to remove the scheduled delay between frames entirely")
    parser.add_argument("-l", "--headless", action="store_true", help="run without rendering the camera feed")
    parser.add_argument("-c", "--calibration", help="file containing calibration parameters")
    args = parser.parse_args()
    args.port_number = int(args.port_number) if str.isdigit(args.port_number) else args.port_number
    print(args)

    calibration_params = None
    if args.calibration is not None:
        calibration_params = load_calibration_params(args.calibration)

    fps = display(args.port_number, fps=args.fps, render=not args.headless, calibration_params=calibration_params)
    print(f"Measured fps was {fps:.2f}")
