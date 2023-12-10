# Kevin David Ruiz Gonzalez

# Ejercicio 3
# Hacer una función en Python que reciba una imagen, un número real, w, entre -1 y
# 1 y un número real, σ que representa la varianza de una función gaussiana. La
# función debe devolver la imagen modificada I′ usando lo siguiente:
# I' = (1 + w) ∙ I − w ∙ I ⋆ Gσ
# Contestar las siguientes preguntas:
# 1. Qué valores de da el máximo blurring. Explicar la razón.
# 2. Qué valores de da el máximo sharpening. Explicar la razón.
# 3. Correr el filtro en cualquier imagen con el máximo valor de blurring (el obtenido
# en la pregunta 1) con σ = 10. Ahora intentar revertir el efecto aplicando el filtro
# con el máximo valor de sharpening (el obtenido en la pregunta 2) con σ = 10.
# ¿Se recuperó la imagen original? ¿Por qué si o por qué no? 

import numpy as np
import cv2

def apply_unsharp_mask(image, w, sigma):
    """
    Applies an unsharp mask to a grayscale 8-bit image.

    Args:
        image (numpy.ndarray): Grayscale 8-bit image.
        w (float): Unsharp mask parameter.
        sigma (float): Standard deviation of Gaussian kernel.

    Returns:
        numpy.ndarray: Image with unsharp mask applied.
    """
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)

    # Apply unsharp mask
    sharpened_image = cv2.addWeighted(image, 1 + w, blurred_image, -w, 0)
    
    # Be sure that the resulting image is in the range [0, 255]
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image

def main():
    """
    Main function.
    """
    # Load image
    image = cv2.imread("red_panda.jpg", cv2.IMREAD_GRAYSCALE)

    # Apply unsharp mask
    w = 0.5
    sigma = 10
    sharpened_image = apply_unsharp_mask(image, w, sigma)

    # Show images
    cv2.imshow("Original", image)
    cv2.imshow("Sharpened", sharpened_image)
    cv2.imwrite("imgs Ej3/Sharpened.jpg", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# 1. Qué valores de da el máximo blurring. Explicar la razón.
# R: El máximo blurring se obtiene con w = 1 y σ = 10. Esto se debe a que el
#    parámetro w controla la cantidad de blurring y el parámetro σ controla la
#    cantidad de desenfoque gaussiano. Al tener w = 1, se aplica el máximo
#    blurring posible, mientras que σ = 10 hace que el desenfoque gaussiano sea
#    muy grande, haciendo que la imagen se vea muy borrosa.

# 2. Qué valores de da el máximo sharpening. Explicar la razón.
# R: El máximo sharpening se obtiene con w = 1 y σ = 0. Esto se debe a que el
#    parámetro w controla la cantidad de sharpening y el parámetro σ controla la
#    cantidad de desenfoque gaussiano. Al tener w = 1, se aplica el máximo
#    sharpening posible, mientras que σ = 0 hace que el desenfoque gaussiano sea
#    nulo, haciendo que la imagen se vea muy nítida.

# 3. Correr el filtro en cualquier imagen con el máximo valor de blurring (el obtenido
#    en la pregunta 1) con σ = 10. Ahora intentar revertir el efecto aplicando el filtro
#    con el máximo valor de sharpening (el obtenido en la pregunta 2) con σ = 10.
#    ¿Se recuperó la imagen original? ¿Por qué si o por qué no?
# R: No se recupera la imagen original. Esto se debe a que el filtro de unsharp
#    mask no es invertible, por lo que no se puede recuperar la imagen original
#    aplicando el filtro de unsharp mask con parámetros opuestos.



