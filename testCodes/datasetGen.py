import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from PIL import Image

data_dir = 'F:/Work/Bachelor-thesis/Data/data2021_ori/90_ori'

if __name__ == "__main__":
    datasheet = pd.read_excel('F:/Work/Bachelor-thesis/Data/datasheet.xlsx')
    testing_index = open(data_dir + "_testing_index.txt", "w")
    training_index = open(data_dir + "_training_index.txt", "w")
    for i in range(1, 23):
        for filename in os.listdir(data_dir + '/' + str(i)):
            img = cv2.imread(data_dir + '/' + str(i) + '/' + filename)
            # plt.figure("originalImage")
            # plt.imshow(img)

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

            img_back[0:30, ] = 255
            img_back[(rows - 30):rows, ] = 255

            plt.figure("img_back")
            plt.imshow(img_back)
            plt.axis('off')
            plt.savefig('F:/Work/Bachelor-thesis/Data/data_highfreq/' + filename, transparent=True)

            # imsave = Image.fromarray(np.uint8(img_back))
            # imsave.save('F:/Work/Bachelor-thesis/Data/data_highfreq/' + filename)

            if random.random() < 0.8:
                training_index.write(filename + '\t' + str(datasheet.loc[i - 1, 'focus']) + '\t' + str(
                    datasheet.loc[i - 1, 'pressure']) + '\n')
            else:
                testing_index.write(filename + '\t' + str(datasheet.loc[i - 1, 'focus']) + '\t' + str(
                    datasheet.loc[i - 1, 'pressure']) + '\n')

            print(filename)
