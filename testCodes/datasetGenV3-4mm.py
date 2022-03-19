import os
import random

import cv2
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

data_dir = 'Data/data2022-4mm/'


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
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # se = cv2.morphologyEx(se, cv2.MORPH_CLOSE, (2, 2))
    # mask = cv2.dilate(binary, se)
    # edge = cv2.Canny(binary, 80, 100)
    # edge = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    # cv2.imshow('edge', edge)
    # cv2.waitKey(0)

    # src_h, src_w = gray.shape[:2]
    # size = max(src_h, src_w)
    dct_size = (720, 720)

    result_original = resize_keep_aspectratio(img, dct_size)
    result_gray = resize_keep_aspectratio(gray, dct_size)
    # cv2.imshow('original', result_original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # result_original = img
    # result_gray = gray

    gblur = cv2.GaussianBlur(result_gray, (5, 5), 0)
    # cv2.imshow('gaussian', gblur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    canny = cv2.Canny(gblur, 80, 300)
    cv2.imshow('canny', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sure = cv2.dilate(canny, kernelX, iterations=2)
    cv2.imshow('sure1', sure)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    opening = cv2.erode(sure, kernelX, iterations=2)
    cv2.imshow('open1', opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    sure2 = cv2.dilate(opening, kernelY, iterations=4)
    cv2.imshow('sure2', sure)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    opening2 = cv2.erode(sure2, kernelY, iterations=2)
    cv2.imshow('opening2', opening2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cover = cv2.dilate(opening2, kernelY, iterations=1)
    cv2.imshow('cover', cover)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    (minX, minY) = (np.inf, np.inf)
    (maxX, maxY) = (-np.inf, -np.inf)
    contours, hier = cv2.findContours(cover.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        minX = min(minX, x)
        minY = min(minY, y)
        maxX = max(maxX, x + w - 1)
        maxY = max(maxY, y + h - 1)

    result_original[0: y, :] = 0
    result_original[y + h:, :] = 0
    result_original[:, 0: x] = 0
    result_original[:, x + w:] = 0
    cropped = result_original

    cv2.imshow('cropped', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)
    # result = cv2.bitwise_and(img, mask)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    return cropped


if __name__ == "__main__":
    datasheet_path = os.path.abspath('Data/4mm.xlsx')
    datasheet = pd.read_excel(datasheet_path, header=0)
    '''
    datasheet['speed'].replace(6, '0', inplace=True)
    datasheet['speed'].replace(7.5, '1', inplace=True)
    datasheet['speed'].replace(9, '2', inplace=True)
    datasheet['speed'].replace(10.5, '3', inplace=True)
    datasheet['speed'].replace(12, '4', inplace=True)

    datasheet.replace(-2, '0', inplace=True)
    datasheet.replace(-2.8, '1', inplace=True)
    datasheet.replace(-3.5, '2', inplace=True)
    datasheet.replace(-4.3, '3', inplace=True)
    datasheet.replace(-5, '4', inplace=True)

    datasheet['pressure'].replace(7, '0', inplace=True)
    datasheet['pressure'].replace(7.8, '1', inplace=True)
    datasheet['pressure'].replace(8.5, '2', inplace=True)
    datasheet['pressure'].replace(9.3, '3', inplace=True)
    datasheet['pressure'].replace(10, '4', inplace=True)
    '''

    testing_index = open("Data/testing_index.txt", "w")
    training_index = open("Data/training_index.txt", "w")

    for filename in os.listdir(data_dir):
        filename = '15.jpg'
        img = cv2.imread(data_dir + '/' + filename)
        # plt.figure("originalImage")
        # plt.imshow(img)
        i = int(filename[0:filename.index('.')])
        savefig = image_reshape(img)

        cv2.imwrite('Data/4mm_processed/' + filename, savefig)
        # imsave = Image.fromarray(np.uint8(img_back))
        # imsave.save('F:/Work/Bachelor-thesis/Data/data_highfreq/' + filename)

        if random.random() < 0.75:
            training_index.write(filename + '\t' + str(datasheet.loc[i - 1, 'speed']) + '\t' + str(
                datasheet.loc[i - 1, 'focus']) + '\t' + str(datasheet.loc[i - 1, 'quality'] - 1) + '\n')
        else:
            testing_index.write(filename + '\t' + str(datasheet.loc[i - 1, 'speed']) + '\t' + str(
                datasheet.loc[i - 1, 'focus']) + '\t' + str(datasheet.loc[i - 1, 'quality'] - 1) + '\n')

        print(filename)
