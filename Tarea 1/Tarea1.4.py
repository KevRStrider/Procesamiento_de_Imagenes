# Kevin David Ruiz Gonzalez

# 4. En el directorio de la tarea hay 2 imágenes: checkerboard.png y dark_street.jpg. El
# ejercicio es desarrollar un programa en Python que para cada imagen haga lo
# siguiente:

# a. Lea la imagen.
# b. Escriba en un archivo el histograma original.
# c. Le aplique el método de ecualización del histograma.
# d. Escriba en un archivo el histograma ecualizado.
# e. Le aplique a la imagen original el método CLAHE.
# f. Escriba en un archivo el histograma modificado.
# g. Analicen si hay diferencia entre los 2 métodos.


import cv2
import numpy as np
from matplotlib import pyplot as plt

METHODS = [
    {"name": "original", "apply": lambda img: img},
    {"name": "equalize", "apply": cv2.equalizeHist},
    {"name": "clahe", "apply": lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)},
]

def save_histogram(hist, filename):
    plt.bar(np.arange(256), hist.flatten())
    plt.xlabel('Valor de píxel')
    plt.ylabel('Frecuencia')
    plt.title('Histograma')
    plt.savefig(filename)
    plt.close()

def process_image(image_path):
    print(f"Procesando imagen: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    for method in METHODS:
        method_name = method["name"]
        processed_image = method["apply"](image)
        hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        save_histogram(hist, f'histogram_{method_name}_{image_path[:-4]}.png')

    print("Procesamiento completo.\n")

def main():
    images = ['checkerboard.png', 'dark_street.jpg']
    
    for image_path in images:
        process_image(image_path)

if __name__ == "__main__":
    main()
