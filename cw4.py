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

# find magnitude & phase
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


# cast data type to float 32 bit
padded_sobel = padded_sobel.astype(np.float32);

# take fourier transform
sobelF = np.fft.fft2(padded_sobel)

# ship (0,0) to center of image
sobelF = np.fft.fftshift(sobelF)

# find magnitude & phase``
sobelReal = np.real(sobelF)
sobelIma = np.imag(sobelF)
sobelMag = np.sqrt(sobelReal**2 + sobelIma**2)
sobelPhs = np.arctan2(sobelIma, sobelReal)

# Inverse Fourier transform
sobelRealInv = sobelMag*np.cos(sobelPhs)
sobelImaInv = sobelMag*np.sin(sobelPhs)

sobelFInv = sobelRealInv + sobelImaInv*1j

sobelFInv = np.fft.ifftshift(sobelFInv)
sobelInv = np.fft.ifft2(sobelFInv)

sobelInv = np.real(sobelInv)
sobelInv = sobelInv.astype(np.uint8);

# display magnitude
#sobelMag = np.log(1+sobelMag)
sobelMag = cv.normalize(sobelMag, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

cv.imwrite("f_sobel.png",sobelMag)

########################## นำรูป F sobel มา คูณ จุดต่อจุด กับ frequency domain image

# img_frequency_domain_sobeled = np.dot(imgMag,sobel_magnitude_image)
# cv.imwrite("f_Fdomain_sobeled.png",img_frequency_domain_sobeled)
img_frequency_domain_sobeled=np.zeros(imgMag.shape)
for y in range(0,imgMag.shape[0]):
    for x in range(0,imgMag.shape[1]):
        img_frequency_domain_sobeled[y,x]= imgMag[y,x] * sobelMag[y,x]




##############################เเปลง f_Fdomain_sobeled.png ให้กลับเป็นรูปตกกะปิ ก็เป็นอันเส็ดสิ้น

fft_image = np.fft.fftshift(img_frequency_domain_sobeled)

# Calculate the inverse Fourier transform
inverse_fft_image = np.fft.ifft2(fft_image)

# Convert the inverse Fourier transform to an image
inverse_fft_image = np.real(inverse_fft_image)

# Normalize the image
inverse_fft_image = np.clip(inverse_fft_image, 0, 255)
inverse_fft_image = np.uint8(inverse_fft_image)

# Save the image
cv.imwrite('f_output.jpg', inverse_fft_image)