import cv2
import numpy as np
img1 = cv2.imread('track1.jpg')
img2 = cv2.imread('track2.jpg')
img3 = cv2.imread('track3.jpg')
img4 = cv2.imread('track5.jpg')
img = cv2.imread('track.jpg')
# cv2.imshow("track1",img1)
# cv2.waitKey(0)


hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv2.imshow("track1HSV",hsv1)
# cv2.waitKey(0)


lower = np.array([0, 155, 155])
upper = np.array([60, 255, 255])
mask = cv2.inRange(hsv1, lower, upper)
# cv2.imshow("track1MASK",mask)
# cv2.waitKey(0)

edges = cv2.Canny(mask, 200, 400)
# cv2.imshow("edges",edges)
# cv2.waitKey(0)

# def detect_line_segments(edges)
# tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
rho = 10  # distance precision in pixel, i.e. 1 pixel
angle = (np.pi / 180)*3  # angular precision in radian, i.e. 1 degree
min_threshold = 10  # minimal of votes
line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold,np.array([]), minLineLength=8, maxLineGap=4)

# return line_segments
print(line_segments)
