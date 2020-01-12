import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def canny(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 155, 155])
    upper = np.array([60, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    canny = cv2.Canny(mask, 50, 150)
    return canny

def region_of_interest(img):
    height = image.shape[0]
    width = image.shape[1]
    # polygons = np.array([[(200,height),(1100,height),(550,250)]])
    polygons = np.array([[
        (125, 100),
        (75, 100),(0,150),(0,height),(width,height),(width,150)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,color=(255,255,255))
    masked_img = cv2.bitwise_and(img,mask)
    return masked_img

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2),(0,255,0,),10)
    return line_image


def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    lane_lines= []
    for line in lines:
            x1,y1,x2,y2= line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2),1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope<-25*np.pi/180:
                left_fit.append((slope, intercept))
            elif slope>25*np.pi/180:
                right_fit.append((slope,intercept))
    left_fit_avg = np.average(left_fit,axis=0)
    if len(left_fit)> 0:
        left_line= make_coordinates(image,left_fit_avg)
        lane_lines.append(left_line)
    right_fit_avg = np.average(right_fit,axis=0)
    if len(right_fit)>0:
        right_line = make_coordinates(image,right_fit_avg)
        lane_lines.append(right_line)
    # mean_line = make_coordinates(image,(left_fit_avg+right_fit_avg)/2)
    # print(lane_lines)
    return lane_lines


image = cv2.imread('track.jpg')
lane_img = np.copy(image)
canny_image = canny(lane_img)
cropped_img = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_img,20,np.pi/180,50,np.array([]),minLineLength=10,maxLineGap=50)
averaged_lines = average_slope_intercept(lane_img,lines)
line_image = display_lines(lane_img,averaged_lines)
overlay_img = cv2.addWeighted(lane_img,0.8,line_image,1,1)
cv2.imwrite("identified_lane.jpg",overlay_img)
cv2.imshow("result", overlay_img)
cv2.waitKey(0)



# following code is allows us to segregate the region of interest.
# plt.imshow(canny_image)
# plt.show()