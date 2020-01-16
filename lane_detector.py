import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 70, 150)
    return canny

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([[(0,400),(width,400),(width,height),(0,height)]])
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
    y2 = int(y1*(1/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    if type(lines)!=type(None):
        for line in lines:
                x1,y1,x2,y2= line.reshape(4)
                parameters = np.polyfit((x1,x2),(y1,y2),1)
                slope=parameters[0]
                intercept=parameters[1]
                if slope<0:
                    left_fit.append((slope, intercept))
                elif 0<slope<85:
                    right_fit.append((slope,intercept))
        left_fit_avg = np.average(left_fit,axis=0)
        right_fit_avg = np.average(right_fit,axis=0)
        left_line= make_coordinates(image,left_fit_avg)
        right_line = make_coordinates(image,right_fit_avg)
        # mean_line = make_coordinates(image,(left_fit_avg+right_fit_avg)/2)
        # print(mean_line)
        return np.array([left_line,right_line])
    else:
        return None;


# image = cv2.imread('capture1.jpg')
# lane_img = np.copy(image)
# canny_image = canny(lane_img)
# cropped_img = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_img,lines)
# line_image = display_lines(lane_img,averaged_lines)
# overlay_img = cv2.addWeighted(lane_img,0.8,line_image,1,1)
# cv2.imshow("result", overlay_img)
# cv2.waitKey(0)


# following code is allows us to segregate the region of interest.
# plt.imshow(canny_image)
# plt.show()


# video section
cap = cv2.VideoCapture("t1.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_img = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_img, 9, 2*np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    if type(averaged_lines)!=type(None):
        line_image = display_lines(frame, averaged_lines)
        overlay_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", overlay_img)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()