# Kevin David Ruiz Gonzalez

# Ejercicio 1
# Programar en Python una función para agregar ruido sal y pimienta a una imagen
# en tonos de grises de 8 bits. Los argumentos de la función son una imagen y un
# número entre 0 y 1 que representa la probabilidad de que un pixel cambie de color
# a blanco (255, ruido sal) o a negro (0, ruido pimienta). Si el pixel se va a convertir
# en ruido, la probabilidad de ser sal o pimienta es la misma. La función debe
# regresar la imagen modificada con ruido.
# Utilizando dicha función, agregar ruido a la imagen “lady-hat.jpg” (o a cualquier
# otra de tamaño similar), primero con una probabilidad de 10% y luego con 25%.
# Para cada imagen con ruido, tratar de eliminar el ruido usando a) un filtro
# gaussiano de 5 x 5 con sigma = 1, b) un filtro gaussiano de 11 x 11 con sigma = 3
# y c) un filtro de mediana.
# Por último, comparar las imágenes obtenidas y ver qué filtro obtuvo mejores
# resultados. 


import numpy as np
import cv2

def add_salt_and_pepper_noise(image, prob):
    """
    Adds salt and pepper noise to a grayscale 8-bit image.

    Args:
        image (numpy.ndarray): Grayscale 8-bit image.
        prob (float): Probability of a pixel changing to salt or pepper noise.

    Returns:
        numpy.ndarray: Image with salt and pepper noise.
    """
    h, w = image.shape[:2]
    noise = np.zeros((h, w), np.uint8)
    cv2.randu(noise, 0, 255)
    salt = (noise > 255 * (1 - prob / 2))
    pepper = (noise < 255 * prob / 2)
    image[salt] = 255
    image[pepper] = 0
    return image

def main():
    """
    Main function.
    """
    image = cv2.imread("lady-hat.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = add_salt_and_pepper_noise(image, 0.1)
    cv2.imshow("Salt and Pepper Noise (10%)", image)
    cv2.waitKey(0)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (10%).jpg", image)
    cv2.destroyAllWindows()
    image = cv2.GaussianBlur(image, (5, 5), 1)
    cv2.imshow("Salt and Pepper Noise (10%) + Gaussian Blur (5 x 5, sigma = 1)", image)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (10%) + Gaussian Blur (5 x 5, sigma = 1).jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.GaussianBlur(image, (11, 11), 3)
    cv2.imshow("Salt and Pepper Noise (10%) + Gaussian Blur (11 x 11, sigma = 3)", image)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (10%) + Gaussian Blur (11 x 11, sigma = 3).jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.medianBlur(image, 5)
    cv2.imshow("Salt and Pepper Noise (10%) + Median Blur (5 x 5)", image)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (10%) + Median Blur (5 x 5).jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.imread("lady-hat.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = add_salt_and_pepper_noise(image, 0.25)
    cv2.imshow("Salt and Pepper Noise (25%)", image)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (25%).jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.GaussianBlur(image, (5, 5), 1)
    cv2.imshow("Salt and Pepper Noise (25%) + Gaussian Blur (5 x 5, sigma = 1)", image)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (25%) + Gaussian Blur (5 x 5, sigma = 1).jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.GaussianBlur(image, (11, 11), 3)
    cv2.imshow("Salt and Pepper Noise (25%) + Gaussian Blur (11 x 11, sigma = 3)", image)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (25%) + Gaussian Blur (11 x 11, sigma = 3).jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.medianBlur(image, 5)
    cv2.imshow("Salt and Pepper Noise (25%) + Median Blur (5 x 5)", image)
    cv2.imwrite("imgs Ej1/Salt and Pepper Noise (25%) + Median Blur (5 x 5).jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()

# Conclusiones
# El filtro que obtuvo mejores resultados fue el filtro de mediana, ya que este filtro
# elimina el ruido sin afectar demasiado la imagen original, a diferencia de los
# filtros gaussianos, que eliminan el ruido pero también la nitidez de la imagen.

