# Kevin David Ruiz González
import cv2 as cv
import numpy as np
import math

image = cv.imread('./images/hall-1.jpg')
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_edges = cv.Canny(image_gray, threshold1=100, threshold2=200, apertureSize=3)  # Adjusted thresholds
image_hough_std = cv.cvtColor(image_edges, cv.COLOR_GRAY2BGR)
image_hough_p = np.copy(image_hough_std)

# standard Hough
lines = cv.HoughLines(image_edges, rho=1, theta=np.pi / 180, threshold=250)  # Adjusted threshold

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv.line(image_hough_std, pt1, pt2, color=(0, 0, 255), thickness=3, lineType=cv.LINE_AA)

# probabilistic Hough
linesP = cv.HoughLinesP(image_edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=10, maxLineGap=30)  # Adjusted threshold

if linesP is not None:
    for line in linesP[:, 0]:
        cv.line(image_hough_p,
                pt1=(int(line[0]), int(line[1])),
                pt2=(int(line[2]), int(line[3])),
                color=(0, 0, 255), thickness=3, lineType=cv.LINE_AA)

cv.imwrite('hough_std.jpg', image_hough_std)
cv.waitKey(0)
cv.imwrite('hough_p.jpg', image_hough_p)
cv.waitKey(0)