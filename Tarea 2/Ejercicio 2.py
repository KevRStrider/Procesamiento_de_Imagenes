# Kevin David Ruiz Gonzalez


# Ejercicio 2
# Hacer una función en Python para enfocar (sharpen) una imagen en tonos de
# grises de 8 bits utilizando el siguiente kernel:
# 0     |c/4    | 0
# c/4   |1 − c  | c/4
# 0     |c/4    | 0
# Los argumentos de la función son una imagen y c, un número real positivo o
# negativo. La función debe devolver la imagen enfocada.
# Contestar las siguientes preguntas:
# 1. ¿Qué valor o valores de c dan los mejores resultados?
# 2. ¿Qué le pasa a la imagen cuando el parámetro c es muy grande o muy pequeño
# positivo o negativo? 

import numpy as np
import cv2

def sharpen(image, c):
    """
    Sharpens a grayscale 8-bit image.

    Args:
        image (numpy.ndarray): Grayscale 8-bit image.
        c (float): Sharpening parameter.

    Returns:
        numpy.ndarray: Sharpened image.
    """
    kernel = np.array([[0, c / 4, 0], [c / 4, 1 - c, c / 4], [0, c / 4, 0]])
    sharpened_img = cv2.filter2D(image, -1, kernel)
    
    # Be sure that the resulting image is in the range [0, 255]
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    return sharpened_img

def main():
    """
    Main function.
    """
    # Load image
    image = cv2.imread("frog.jpg", cv2.IMREAD_GRAYSCALE)

    # Apply sharpening
    c = -3
    sharpened_image = sharpen(image, c)

    # Show images
    cv2.imshow("Original", image)
    cv2.imshow("Sharpened", sharpened_image)
    cv2.imwrite("imgs Ej2/Sharpened.jpg", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# 1. ¿Qué valor o valores de c dan los mejores resultados?
# R: Los valores de c que dan los mejores resultados son los valores cercanos a -1.

# 2. ¿Qué le pasa a la imagen cuando el parámetro c es muy grande o muy pequeño, positivo o negativo?
#    - Cuando c es muy grande (>1), se invierte el efecto de enfoque, dando lugar a una imagen suavizada.
#    - Cuando c es muy pequeño (cercano a 0), el efecto de enfoque es mínimo o nulo.
#    - Cuando c es negativo el efecto de enfoque es muy fuerte, con bordes exagerados. 




