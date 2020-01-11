import cv2
import numpy as np

def detect_edges(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    lower_orange = np.array([0, 155, 155])
    upper_orange = np.array([60, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    cv2.imshow("Orange mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges

