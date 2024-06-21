import cv2
import numpy as np
import argparse

# def find_and_draw_contours(frame):
#     cpy = frame.copy()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
#     # edges = cv2.Canny(thresh, 100, 200)
#     # thresh_lines_removed = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
#     thresh_lines_removed = thresh.copy()
#     lines = cv2.HoughLinesP(thresh, 1, np.pi / 2, 2, None, 40, 5)
#     if lines is not None:
#         for line in lines[:, 0, :]:
#             pt1 = (line[0],line[1])
#             pt2 = (line[2],line[3])
#             cv2.line(thresh_lines_removed, pt1, pt2, (0,0,0), 5)
    
#     # Apply morphological operations to remove small noise and lines
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     morphed = cv2.morphologyEx(thresh_lines_removed, cv2.MORPH_OPEN, kernel)
#     # morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    
#     contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area

#     # Define minimum and maximum area for contours to be considered fish
#     min_area = 40
#     max_area = 1000
#     border_margin = 0  # Margin from the border to ignore contours

#     height, width = frame.shape[:2]

#     for contour in contours[:3]:
#         area = cv2.contourArea(contour)
#         if min_area < area < max_area:
#             # Calculate the bounding rectangle for the contour
#             x, y, w, h = cv2.boundingRect(contour)
#             aspect_ratio = w / float(h)
            
#             # Filter out contours that are extremely narrow or tall
#             if 0.2 < aspect_ratio < 5.0:
#                 # Filter out contours that are close to the border
#                 if x > border_margin and y > border_margin and (x + w) < (width - border_margin) and (y + h) < (height - border_margin):
#                     # Draw the contour on the frame
#                     cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    
#                     # Calculate the centroid of the contour and draw it
#                     M = cv2.moments(contour)
#                     if M["m00"] != 0:
#                         cX = int(M["m10"] / M["m00"])
#                         cY = int(M["m01"] / M["m00"])
#                         cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

#     # return cpy
#     # return gray
#     # return thresh
#     # return edges
#     # return thresh_lines_removed
#     # return morphed
#     return frame

def find_and_draw_contours(frame):
    cpy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 40)

    # Apply morphological operations to remove small noise and lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area

    # Define minimum and maximum area for contours to be considered fish
    min_area = 0
    max_area = 1000
    border_margin = 0  # Margin from the border to ignore contours

    height, width = frame.shape[:2]

    for contour in contours[:2]:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filter out contours that are extremely narrow or tall
            if 0.2 < aspect_ratio < 5.0:
                # Filter out contours that are close to the border
                if x > border_margin and y > border_margin and (x + w) < (width - border_margin) and (y + h) < (height - border_margin):
                    # Draw the contour on the frame
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    
                    # Calculate the centroid of the contour and draw it
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

    # return cpy
    # return gray
    # return thresh
    # return morphed
    return frame

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    last_frame_with_contours = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_contours = find_and_draw_contours(frame)
        cv2.imshow('Frame with Contours', frame_with_contours)
        # if last_frame_with_contours is not None:
        #     cv2.imshow('Frame with Contours', cv2.absdiff(last_frame_with_contours, frame_with_contours))
        last_frame_with_contours = frame_with_contours

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video to find and draw contours.')
    parser.add_argument('video', type=str, help='Path to the input video file')
    args = parser.parse_args()

    main(args.video)
