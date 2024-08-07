import cv2
import numpy as np
import argparse
from centroidTracker import CentroidTracker
try:
    from .calibration import load_calibration_params, apply_calibration, undistort_and_warp
except:
    from calibration import load_calibration_params, apply_calibration, undistort_and_warp

def find_and_draw_contours(frame, centroid_tracker, initial_objects):
    cpy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    thresh_lines_removed = thresh.copy()
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 2, 2, None, 40, 5)
    if lines is not None:
        for line in lines[:, 0, :]:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(thresh_lines_removed, pt1, pt2, (0, 0, 0), 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh_lines_removed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = 40
    max_area = 1000
    border_margin = 0

    height, width = frame.shape[:2]
    input_centroids = []

    for contour in contours[:initial_objects]:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if 0.2 < aspect_ratio < 5.0:
                if x > border_margin and y > border_margin and (x + w) < (width - border_margin) and (y + h) < (height - border_margin):
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        input_centroids.append((cX, cY))
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

    # print(input_centroids)
    objects = centroid_tracker.update(input_centroids)

    # print("new frame")
    # print(len(input_centroids), objects.items())
    for objectID, centroid in objects.items():
        text = f"ID {objectID}"
        centroid = np.array(centroid, dtype=int)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return frame

def main(port_number, calibration_params, initial_objects):
    cap = cv2.VideoCapture(port_number)
    centroid_tracker = CentroidTracker(initial_objects)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if calibration_params is not None:
            frame = apply_calibration(frame, calibration_params)

        frame_with_contours = find_and_draw_contours(frame, centroid_tracker, initial_objects)
        cv2.imshow("Frame with Contours", frame_with_contours)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to find and draw contours.")
    parser.add_argument("port_number", help="Path to the input video")
    parser.add_argument("-n", "--num_objects", type=int, default=0, help="Number of objects to track")
    parser.add_argument("-c", "--calibration", help="file containing calibration parameters")
    args = parser.parse_args()
    args.port_number = int(args.port_number) if str.isdigit(args.port_number) else args.port_number

    calibration_params = None
    if args.calibration is not None:
        calibration_params = load_calibration_params(args.calibration)

    main(args.port_number, calibration_params, args.num_objects)
