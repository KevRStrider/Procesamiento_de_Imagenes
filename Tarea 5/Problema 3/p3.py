# Kevin David Ruiz González 

# Image matching
# Seleccionar dos imágenes del directorio “images/image-matching”, encontrar los
# mejores matches y dibujarlos con línea verdes.
# Este ejercicio se puede resolver mediante alguno de los dos métodos vistos en
# clase:
# a) método knnMatch de la clase BFMatcher.
# b) Método knnMatch de la clase FlannBasedMatcher.
# Con cualquiera de las dos opciones se debe usar el ratio test (ratio = 0.7 o 0.8)
# para filtrar los resultados. Los features se pueden obtener usando SIFT u ORB. 

# Decidí usar las imágenes p1.jpg y p2.jgp de eynsham, estas imágenes las dejé en la carpeta images 

import cv2
import numpy as np

# Función para redimensionar las imágenes manteniendo la proporción
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# Cargar las imágenes y redimensionarlas
img1 = cv2.imread('images/p1.jpg', 0)
img2 = cv2.imread('images/p2.jpg', 0)

scale_percent = 50  # Por ejemplo, se reduce a la mitad
img1 = resize_image(img1, scale_percent)
img2 = resize_image(img2, scale_percent)

# Inicializar el detector de características (SIFT o ORB)
sift = cv2.SIFT_create()
# orb = cv2.ORB_create()

# Encontrar los keypoints y descriptores de cada imagen
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# Inicializar el matcher BF
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)  # k=2 para obtener los 2 mejores matches para cada descriptor

# Ratio test para filtrar los matches buenos
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  
        good_matches.append(m)

# Dibujar solo las líneas verdes que conectan los matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))

# Mostrar la imagen con las líneas verdes que conectan los matches
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen de salida
cv2.imwrite('matches.png', img_matches)
