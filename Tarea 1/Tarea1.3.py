# Kevin David Ruiz Gonzalez

# 3. Desarrollar un programa en Python que transforme una imagen a blanco y negro
# usando la mediana de las intensidades como umbral.

import cv2

# Carga la imagen
imagen = cv2.imread('img2.jpeg')

# Reajusta la imagen a 500x500 píxeles
imagen_reajustada = cv2.resize(imagen, (500, 500))

# Convierte la imagen reajustada a escala de grises
imagen_gris = cv2.cvtColor(imagen_reajustada, cv2.COLOR_BGR2GRAY)

# Calcula la mediana de las intensidades en la imagen en escala de grises
mediana = cv2.medianBlur(imagen_gris, 5)  # El valor 5 es el tamaño del kernel

# Aplica un umbral adaptativo a la imagen utilizando la mediana como valor de umbral
imagen_umbral = cv2.adaptiveThreshold(mediana, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Muestra la imagen original reajustada y la imagen en blanco y negro
cv2.imshow('Imagen Original Reajustada', imagen_reajustada)
cv2.imshow('Imagen Blanco y Negro', imagen_umbral)

# Espera a que el usuario presione una tecla y luego cierra las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()

# Escribir las imagenes
cv2.imwrite('img_2blancoynegro.jpg', imagen_umbral)

