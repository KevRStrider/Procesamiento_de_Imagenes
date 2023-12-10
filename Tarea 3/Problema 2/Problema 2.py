# Kevin David Ruiz González
import cv2
import numpy as np 

# Load images
original_image = cv2.imread("./images/Lahore-Fort.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.imread("./images/Lahore-Fort-edges.jpg", cv2.IMREAD_GRAYSCALE)

# Resize edges to match original_image
edges = cv2.resize(edges, (796, 600))

# Apply Canny edge detection
canny_filtered = cv2.Canny(original_image, threshold1=0, threshold2=500, apertureSize=3)

# Concatenate images for comparison
comparation = cv2.hconcat([original_image, edges, canny_filtered])

# Save the comparison image
cv2.imwrite("Comparativa.jpg", comparation)

# Calculate differences
differences = np.square(canny_filtered - edges)
difference_sum = np.sum(differences)
zero_percentage = (np.count_nonzero(differences == 0) / differences.size) * 100

# Print results
print("Diferencia:" + str(difference_sum) + "\n")
print("Porcentaje de ceros:" + str(zero_percentage) + "\n")



