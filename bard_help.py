import cv2 as cv
import numpy as np

# Load the image
image = cv.imread("demo.png",cv.IMREAD_GRAYSCALE)

# Get the original width and height of the image
width = image.shape[1]
height = image.shape[0]

# Get the desired width and height of the padded image
new_width = 500
new_height = 200

# Create a blank image of the desired size
padded_image = np.zeros((new_height, new_width), dtype=np.uint8)

# Copy the original image into the center of the padded image
top = (new_height - height) // 2
left = (new_width - width) // 2
padded_image[top : top+height, left : left+width] = image

# Display the padded image
cv.imshow("Padded Image", padded_image)
cv.waitKey()
cv.destroyAllWindows()