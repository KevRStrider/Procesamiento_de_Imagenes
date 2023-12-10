# Kevin David Ruiz Gonzalez

# a) Aplicar el método de template matching para encontrar la imagen “coke-logo.jpg” en la imagen “coke-bottle.jpg”, dibujando el resultado con un
# rectángulo verde. Las imagenes se encuentran en la carpeta “images”.

# Importar librerías
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Leer las imágenes
image = cv.imread('images/coke-bottle.jpg')
template = cv.imread('images/coke-logo.jpg')

# Convertir a escala de grises
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

# Obtener las dimensiones del template
height, width = gray_template.shape

# Aplicar template matching
result = cv.matchTemplate(image=gray_image, templ=gray_template, method=cv.TM_CCOEFF_NORMED)

# Obtener los valores de coincidencia
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

# Dibujar un rectángulo alrededor del área coincidente
top_left = max_loc
bottom_right = (top_left[0] + width, top_left[1] + height)
cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Mostrar la imagen de salida
cv.imshow('Final image', image)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('template-matching.png', image)


# b) Aplicar el método de multi-template matching con y sin supresión no-máxima
# (NMS) para encontrar todas las ocurrencias de la imagen “g-logo.png” en la
# imagen “g-image.png”, dibujando los resultados con rectángulos verdes.
# Escribir un párrafo comparando el número de matches sin usar NMS contra si
# usarlo. 

import cv2 as cv
import numpy as np

# Cargar las imágenes
img = cv.imread('images/g-image.png')
template = cv.imread('images/g-logo.png')
h, w = template.shape[:2]

# Aplicar multi-template matching sin NMS
res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
threshold = 0.8  # Umbral de coincidencia

loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

# Aplicar multi-template matching con NMS
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
threshold = 0.8  # Umbral de coincidencia
matches = []
while max_val >= threshold:
    matches.append(max_loc)
    cv.rectangle(res, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), -1)
    cv.floodFill(res, None, max_loc, (0, 0, 0))
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

for match in matches:
    cv.rectangle(img, match, (match[0] + w, match[1] + h), (0, 255, 0), 2)

# Mostrar resultados
cv.imshow('Matches without NMS', img)
cv.imshow('Matches with NMS', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('multi-template-matching-without-nms.png', img)
cv.imwrite('multi-template-matching-with-nms.png', img)


