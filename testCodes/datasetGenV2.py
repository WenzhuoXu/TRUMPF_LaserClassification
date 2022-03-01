import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir = 'F:/BachelorThesis/Data/data2021_ori/90_ori'


def resize_keep_aspectratio(image_src, dst_size):
    src_h, src_w = image_src.shape[:2]
    # print(src_h, src_w)
    dst_h, dst_w = dst_size

    # 判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))

    h_, w_ = image_dst.shape[:2]
    # print(h_, w_)

    top = int((dst_h - h_) / 2);
    down = int((dst_h - h_ + 1) / 2);
    left = int((dst_w - w_) / 2);
    right = int((dst_w - w_ + 1) / 2);

    value = 0
    borderType = cv2.BORDER_CONSTANT
    # print(top, down, left, right)
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)

    return image_dst


def image_reshape(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # se = cv2.morphologyEx(se, cv2.MORPH_CLOSE, (2, 2))
    # mask = cv2.dilate(binary, se)

    src_h, src_w = binary.shape[:2]
    size = max(src_h, src_w)
    dct_size = (size, size)

    result_gray = resize_keep_aspectratio(binary, dct_size)

    result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)
    # result = cv2.bitwise_and(img, mask)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    return result


if __name__ == "__main__":
    datasheet = pd.read_excel('F:/BachelorThesis/Data/datasheet.xlsx')
    datasheet['speed'].replace(6, 0, inplace=True)
    datasheet['speed'].replace(7.5, 1, inplace=True)
    datasheet['speed'].replace(9, 2, inplace=True)
    datasheet['speed'].replace(10.5, 3, inplace=True)
    datasheet['speed'].replace(12, 4, inplace=True)

    datasheet.replace(-2, 0, inplace=True)
    datasheet.replace(-2.8, 1, inplace=True)
    datasheet.replace(-3.5, 2, inplace=True)
    datasheet.replace(-4.3, 3, inplace=True)
    datasheet.replace(-5, 4, inplace=True)

    datasheet['pressure'].replace(7, 0, inplace=True)
    datasheet['pressure'].replace(7.8, 1, inplace=True)
    datasheet['pressure'].replace(8.5, 2, inplace=True)
    datasheet['pressure'].replace(9.3, 3, inplace=True)
    datasheet['pressure'].replace(10, 4, inplace=True)

    testing_index = open(data_dir + "_testing_index.txt", "w")
    training_index = open(data_dir + "_training_index.txt", "w")
    for i in range(1, 23):
        for filename in os.listdir(data_dir + '/' + str(i)):
            img = cv2.imread(data_dir + '/' + str(i) + '/' + filename)
            # plt.figure("originalImage")
            # plt.imshow(img)

            savefig = image_reshape(img)

            cv2.imwrite('F:/BachelorThesis/Data/data_highfreq/' + filename, savefig)
            # imsave = Image.fromarray(np.uint8(img_back))
            # imsave.save('F:/Work/Bachelor-thesis/Data/data_highfreq/' + filename)

            if random.random() < 0.8:
                training_index.write(filename + '\t' + str(datasheet.loc[i - 1, 'speed']) + '\t' + str(
                    datasheet.loc[i - 1, 'focus']) + '\t' + str(
                    datasheet.loc[i - 1, 'pressure']) + '\t' + str(datasheet.loc[i - 1, 'quality'] - 1) + '\n')
            else:
                testing_index.write(filename + '\t' + str(datasheet.loc[i - 1, 'speed']) + '\t' + str(
                    datasheet.loc[i - 1, 'focus']) + '\t' + str(
                    datasheet.loc[i - 1, 'pressure']) + '\t' + str(datasheet.loc[i - 1, 'quality'] - 1) + '\n')

            print(filename)
