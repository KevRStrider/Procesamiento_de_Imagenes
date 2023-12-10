# Kevin David Ruiz González
import cv2
import numpy as np 
import os

def kirsch_filter(gray):
        kernels = [np.array(([5,5,5],[-3,0,-3],[-3,-3,-3]), dtype=np.float32),
                           np.array(([5,5,-3],[5,0,-3],[-3,-3,-3]), dtype=np.float32),
                           np.array(([5,-3,-3],[5,0,-3],[5,-3,-3]), dtype=np.float32),
                           np.array(([-3,-3,-3],[5,0,-3],[5,5,-3]), dtype=np.float32),
                           np.array(([-3,-3,-3],[-3,0,-3],[5,5,5]), dtype=np.float32),
                           np.array(([-3,-3,-3],[-3,0,5],[-3,5,5]), dtype=np.float32),
                           np.array(([-3,-3,5],[-3,0,5],[-3,-3,5]), dtype=np.float32),
                           np.array(([-3,5,5],[-3,0,5],[-3,-3,-3]), dtype=np.float32)]

        images = np.empty((len(kernels),) + gray.shape, dtype=np.float32)

        for i, kernel in enumerate(kernels):
                images[i] = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernel), None, 0, 255, cv2.NORM_MINMAX)

        combined_image = np.max(images, axis = 0)

        return combined_image

image_path = "./images/Lahore-Fort.jpg"

if not os.path.isfile(image_path):
        print(f"Image file does not exist: {image_path}")
else:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if original_image is None:
                print(f"Failed to read image file: {image_path}")
        else:
                filtered = kirsch_filter(original_image)
                filtered = filtered.astype(np.uint8)

                comparation = cv2.hconcat([original_image, filtered])

                cv2.imwrite("result.jpg", comparation)

                differences = (original_image-filtered)**2
                difference_sum = np.sum(differences)
                zero_percentage = (np.count_nonzero(differences == 0)/differences.size)*100

                print("Diferencia: " + str(difference_sum)+"\n")
                print("Porcentaje de ceros: "+str(zero_percentage)+"%")
