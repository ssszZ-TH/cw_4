import cv2
import numpy as np

# Load the image in frequency domain
fft_image = cv2.imread('f_Fdomain_sobeled.png', cv2.IMREAD_GRAYSCALE)

# Shift the zero frequency component to the center of the spectrum
fft_image = np.fft.fftshift(fft_image)

# Calculate the inverse Fourier transform
inverse_fft_image = np.fft.ifft2(fft_image)

# Convert the inverse Fourier transform to an image
inverse_fft_image = np.real(inverse_fft_image)

# Normalize the image
inverse_fft_image = np.clip(inverse_fft_image, 0, 255)
inverse_fft_image = np.uint8(inverse_fft_image)

# Save the image
cv2.imwrite('f_output.jpg', inverse_fft_image)