# Kevin David Ruiz González

# Detección de esquinas
# Comparar el detector de Harris con el detector de Shi-Tomasi. Ejecutar cada
# método en las dos imágenes “images/grace-hopper.png” y "images/sudoku-blank-grid.png",
# dibujando los resultados en verde y escribir un párrafo con su opinión sobre las diferencias. 

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Leer la imagen
img = cv.imread('images/grace-hopper.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Encontrar las esquinas con el detector de Harris
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
dst = cv.dilate(dst, None)
ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

# Encontrar las coordenadas de las esquinas
coordinates = np.argwhere(dst != 0)
coordinates = [coordinate[::-1] for coordinate in coordinates]

# Dibujar un círculo alrededor de cada esquina
for coordinate in coordinates:
    cv.circle(img, tuple(coordinate), 3, (0, 255, 0), 2)

# Mostrar la imagen de salida
cv.imshow('Final image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('harris-grace.png', img)

# Leer la imagen
img = cv.imread('images/grace-hopper.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Encontrar las esquinas con el detector de Shi-Tomasi
corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

# Dibujar un círculo alrededor de cada esquina
for corner in corners:
    x, y = corner.ravel()
    cv.circle(img, (x, y), 3, (0, 255, 0), 2)

# Mostrar la imagen de salida
cv.imshow('Final image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('shi-tomasi-grace.png', img)



####### Sudoku ########
# Leer la imagen
img = cv.imread('images/sudoku-blank-grid.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Encontrar las esquinas con el detector de Harris
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
dst = cv.dilate(dst, None)
ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

# Encontrar las coordenadas de las esquinas
coordinates = np.argwhere(dst != 0)
coordinates = [coordinate[::-1] for coordinate in coordinates]

# Dibujar un círculo alrededor de cada esquina
for coordinate in coordinates:
    cv.circle(img, tuple(coordinate), 3, (0, 255, 0), 2)

# Mostrar la imagen de salida
cv.imshow('Final image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('harris-sudoku.png', img)

# Leer la imagen
img = cv.imread('images/sudoku-blank-grid.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Encontrar las esquinas con el detector de Shi-Tomasi
corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

# Dibujar un círculo alrededor de cada esquina
for corner in corners:
    x, y = corner.ravel()
    cv.circle(img, (x, y), 3, (0, 255, 0), 2)

# Mostrar la imagen de salida
cv.imshow('Final image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('shi-tomasi-sudoku.png', img)

