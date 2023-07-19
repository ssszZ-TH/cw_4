# fourier transformation.py
import numpy as np 
import cv2 as cv

img = cv.imread('input.jpg', cv.IMREAD_GRAYSCALE)

# cast data type to float 32 bit
img = img.astype(np.float32);

# take fourier transform
imgF = np.fft.fft2(img)

# ship (0,0) to center of image
## รูปจริงๆ ที่เป็นสมการ fourier
imgF = np.fft.fftshift(imgF)

# find magnitude & phase``
imgReal = np.real(imgF)
imgIma = np.imag(imgF)
imgMag = np.sqrt(imgReal**2 + imgIma**2)

# เเปลงให้ภาพอ่านรู้เรื่อง
imgMag = np.log(1+imgMag)
imgMag = cv.normalize(imgMag, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
print("input image size y x =",imgMag.shape)
cv.imwrite("f_image.png", imgMag)

##########################สร้าง fourier sobel
sobel_filter = [
    [-1,0,1],
    [-2,0,1],
    [-1,0,1]
]

sobel_filter = np.array(sobel_filter)

#ขนาดเดิม ของ sobel filter
width = sobel_filter.shape[1]
height = sobel_filter.shape[0]

#ขนาด ใหม่ ที่ต้องการ
new_height,new_width = imgMag.shape

# Create a blank image of the desired size
# # filter sobel จะต้องใหญ่ เท่า image size เพื่อให้ทำ dot product ได้
padded_sobel = np.zeros((new_height, new_width), dtype=np.uint8)

# Copy the original image into the center of the padded image
top = (new_height - height) // 2
left = (new_width - width) // 2
padded_sobel[top : top+height, left : left+width] = sobel_filter
## ขนาดจะต้องเท่ากับรูป input เพราะเรา resize เเล้ว 
print("padded sobel y x = ",padded_sobel.shape)

####################### เอา รูป sobel filter convert เป็น frequency domain โดยการทำ fourier tranform

# Take the Fourier transform of the image
sobel_fft_image = np.fft.fft2(padded_sobel)

# Shift the zero frequency component to the center of the spectrum
sobel_fft_image = np.fft.fftshift(sobel_fft_image)

# Calculate the magnitude of the Fourier transform
sobel_magnitude_image = np.abs(sobel_fft_image)

# Save the magnitude image
cv.imwrite('f_sobel.png', sobel_magnitude_image)

img_frequency_domain_sobeled = np.dot(imgMag,sobel_magnitude_image)
cv.imwrite("f_Fdomain_sobeled.png",img_frequency_domain_sobeled)