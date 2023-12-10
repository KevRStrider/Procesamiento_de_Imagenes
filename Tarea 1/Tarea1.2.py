# Kevin David Ruiz Gonzalez

# 2. Desarrollar un programa en Python que calcule el histograma acumulativo de una
# imagen en escala de grises de 8 bits y lo guarde en un archivo. La imagen del
# histograma acumulativo debe ser similar a esta imagen:

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
img = cv2.imread('img2.jpeg')

# Convertir la imagen a escala de grises
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calcular el histograma
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Calcular el histograma acumulativo
hist_acum = np.cumsum(hist)

# Mostrar el histograma acumulativo
plt.plot(hist_acum)
plt.show()

# Guardar el histograma acumulativo como una imagen
plt.savefig('histograma_acumulativo.png')
plt.close()






