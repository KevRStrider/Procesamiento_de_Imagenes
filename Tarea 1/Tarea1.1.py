# 1 Desarrollar un programa en Python que lea dos imágenes, las escale al mismo
# tamaño, tome la mitad izquierda de la primera imagen y la mitad derecha de la
# segunda imagen, y escriba dos nuevas imágenes de la siguiente forma

import cv2
import numpy as np

# Leer las imagenes
img1 = cv2.imread('img1.jpeg')
img2 = cv2.imread('img2.jpeg')

# Escalar las imagenes
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

# Tomar la mitad izquierda de la primera imagen
img_1 = img1[:, :250]

# Tomar la mitad derecha de la segunda imagen
img_2 = img2[:, 250:]

# Concatenar las imagenes
img_12 = np.concatenate((img_1, img_2), axis=1)

# Hacer lo mismo pero con el otro lado de las imagenes
img_1 = img2[:, :250]
img_2 = img1[:, 250:]

img_21 = np.concatenate((img_1, img_2), axis=1)
# Mostrar la imagen
cv2.imshow('Imagen', img_12)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Imagen', img_21)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Escribir las imagenes
cv2.imwrite('img_1.jpg', img_12)
cv2.imwrite('img_2.jpg', img_21)
