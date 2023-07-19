import numpy as np
import cv2 as cv

img = cv.imread("input.jpg",cv.IMREAD_GRAYSCALE)

laplacian = cv.Laplacian(img,cv.CV_64F,ksize=15)
sobelx = cv.Sobel(img, cv.CV_64F,1,0,ksize=15)
sobely = cv.Sobel(img, cv.CV_64F,0,1,ksize=15)

print("input type",img.dtype)
print("laplacian type",laplacian.dtype)
print("sobel x",sobelx.dtype)
print("sobel y",sobely.dtype)

laplacian = cv.normalize(laplacian,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
sobelx = cv.normalize(sobelx,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
sobely = cv.normalize(sobely,None,0,255,cv.NORM_MINMAX,cv.CV_8U)

cv.imwrite("output_laplace.png",laplacian)
cv.imwrite("sobelx.png",sobelx)
cv.imwrite("sobely.png",sobely)