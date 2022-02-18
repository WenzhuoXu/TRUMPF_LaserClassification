import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# from PIL import Image

data_dir = 'F:/Work/Bachelor-thesis/Data/data2021_ori/90_ori'

if __name__ == "__main__":
    with open(data_dir + "index.txt", "w") as index:
        for i in range(1, 23):
            for filename in os.listdir(data_dir + '/' + str(i)):
                img = cv2.imread(data_dir + '/' + str(i) + '/' + filename)
                plt.figure("originalImage")
                plt.imshow(img)

                f = np.fft.fft2(img)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift))

                rows, cols = img.shape[:2]
                crow, ccol = int(rows / 2), int(cols / 2)
                # plt.figure("prefourier")
                # plt.imshow(fshift)
                fshift[(crow - 30):(crow + 30), (ccol - 30):(ccol + 30)] = 0
                # plt.figure("aftfourier")
                # plt.imshow(fshift)
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)

                plt.figure("img_back")
                plt.imshow(img_back)
                plt.axis('off')
                plt.savefig('F:/Work/Bachelor-thesis/Data/data_highfreq/' + filename, transparent=True)
                # imsave = Image.fromarray(np.uint8(img_back))
                # imsave.save('F:/Work/Bachelor-thesis/Data/data_highfreq/' + filename)
                index.write(filename + '\n')
                print(filename)
