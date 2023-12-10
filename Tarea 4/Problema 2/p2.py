# Kevin David Ruiz Gonzalez

# Aplicar la técnica de watershed a la imagen "coins1.png" y "coins2.png" para segmentar las monedas 
# y dibujar un círculo rojo alrededor de su borde 

# COINS1.PNG
import numpy as np
import cv2 as cv

# Leer la imagen
image = cv.imread('coins1.png')

# Convertir a escala de grises
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Binarizar la imagen con un umbral adecuado
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Eliminar ruido con una operación de apertura
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# Identificar el área de fondo seguro
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Transformación de distancia
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

# Umbral para identificar el área de primer plano seguro
_, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# Identificar la región desconocida
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Etiquetado de marcadores
_, markers = cv.connectedComponents(sure_fg)

# Agregar uno a todas las etiquetas para que el fondo seguro no sea 0, sino 1
markers = markers + 1

# Marcar la región desconocida con 0
markers[unknown == 255] = 0

# Aplicar Watershed
cv.watershed(image, markers)

# Encontrar contornos en la imagen binarizada
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Dibujar un círculo alrededor de cada moneda
for contour in contours:
    (x, y, w, h) = cv.boundingRect(contour)
    cv.circle(image, (x + w // 2, y + h // 2), int((w + h) / 4), (0, 0, 255), 2)

# Mostrar la imagen de salida
cv.imshow('Final image', image)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('coins_circles1.png', image)




# COINS2.PNG
import numpy as np
import cv2 as cv

# Leer la imagen
image = cv.imread('coins2.png')

# Convertir a escala de grises
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Aplicar un filtro gaussiano para reducir el ruido
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Binarizar la imagen
_, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Encontrar contornos
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Crear una copia de la imagen para dibujar los contornos
result_image = image.copy()

# Dibujar contornos alrededor de las monedas
for contour in contours:
    if cv.contourArea(contour) > 100:  # Evitar contornos pequeños que no sean monedas
        cv.drawContours(result_image, [contour], 0, (0, 0, 255), 2)

# Mostrar la imagen de salida
cv.imshow('Final image', result_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen de salida
cv.imwrite('coins_circles2.png', result_image)
