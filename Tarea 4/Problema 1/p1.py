# Kevin David Ruiz Gonzalez

# Usando la técnica de histogram backprojection, quitar la pelota de la imagen "fussball-orange.jpgq"

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('fussball-orange.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

roi = image[140:210, 30:150]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

hue, saturation, value = cv2.split(hsv_roi)

roi_hist = cv2.calcHist(images=[hsv_roi], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 180, 0, 256])
cv2.normalize(src=roi_hist, dst=roi_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
mask = cv2.calcBackProject(images=[hsv_image], channels=[0, 1], hist=roi_hist, ranges=[0, 180, 0, 256], scale=1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
imp_mask = cv2.filter2D(mask, -1, kernel)

ret, thresh_mask = cv2.threshold(imp_mask, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

final_mask = cv2.merge((thresh_mask, thresh_mask, thresh_mask))
result = cv2.bitwise_and(image, final_mask)

# Mostrar la imagen de salida
cv2.imshow('Final image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen en archivo
cv2.imwrite('final_image.jpg', result)

