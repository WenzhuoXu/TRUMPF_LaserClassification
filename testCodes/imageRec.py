import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('0326_164430.jpg')
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

rows, cols = img.shape[:2]
crow, ccol = int(rows / 2), int(cols / 2)
fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_back)  # 恢复图像
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.show()
