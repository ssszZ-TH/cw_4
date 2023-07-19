import cv2
import numpy as np

# Load the image
image = cv2.imread("input.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the image to floating point
gray_image = np.float32(gray_image)

# Get the Fourier transform of the image
fft = cv2.dft(gray_image, flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero frequency component to the center of the spectrum
fft_shifted = np.fft.fftshift(fft)

# Calculate the magnitude of the Fourier transform
magnitude = np.abs(fft_shifted)

# Convert the magnitude to an image
magnitude_image = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Inverse Fourier transform the magnitude image
ifft = cv2.idft(magnitude_image)

# Shift the zero frequency component back to the corner of the spectrum
ifft_shifted = np.fft.ifftshift(ifft)

# Get the real part of the inverse Fourier transform
ifft_image = np.real(ifft_shifted)

# Display the original image, the Fourier transform, the magnitude image, and the inverse Fourier transform
cv2.imshow("Original", image)
cv2.waitKey(0)

cv2.imshow("Inverse Fourier", ifft_image)
cv2.waitKey(0)

# Wait for the user to press a key

# Close all windows
cv2.destroyAllWindows()
