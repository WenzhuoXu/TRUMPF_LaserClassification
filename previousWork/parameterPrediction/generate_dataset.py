"""
    This code locate the cutting edge of workpiece, and crop the image into an appropriate size.
    Data Augmentation including rotating, flipping, adjusting contrast ratio and brightness was
    applied to build our custom dataset. The pictures are divided into training set and
    validation set in a ratio of about 8:2.
    Noting that this code only works with the photos shot at 0 degree.
"""

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def measure_object(image):
    """
    This function detects the edge of the workpiece by finding a minimum bounding rectangle.

    Args:
        image:      the image to be detected
    Returns:
        (x, y):     the coordinate of left-top point
        w:          width of the bounding rectangle
        h:          height of the bounding rectangle
    """

    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    plt.figure()
    plt.imshow(th, 'gray')

    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max_area:
            continue
        else:
            max_area = area
            x, y, w, h = cv2.boundingRect(contour)
    plt.plot([x, w], [h, h])
    plt.show()
    return x, y, w, h


def contrast(image):
    """ adjust the contrast ratio of the input image """
    middle = np.zeros(image.shape, image.dtype)
    dst = cv2.addWeighted(image, 1.2, middle, 0, 0)
    return dst


def bright(image):
    """ adjust the brightness of the input image """
    middle = np.zeros(image.shape, image.dtype)
    dst = cv2.addWeighted(image, 1.0, middle, 0, 50)
    return dst


def data_augment(image, name, train_root, val_root, ratio=0.8):
    """
    This function applies data augmentation including rotating, flipping, adjusting
    contrast ratio and brightness(and their mixture) to the given image, and then
    saves the result to the given path.

    Args:
        image:          the source image to be augmented
        name:           the filename which the image will be save as
        train_root:     the output path of the training set
        val_root:       the output path of the validation set
        ratio:          ratio of training set to the whole dataset(it should be a float number between 0-1)
    """

    # instantiate the rotation matrix to rotate the image for -90, 90, and 180 degrees
    matRotate_90 = cv2.getRotationMatrix2D((127, 127), -90, 1.0)
    matRotate90 = cv2.getRotationMatrix2D((127, 127), 90, 1.0)
    matRotate180 = cv2.getRotationMatrix2D((127, 127), 180, 1.0)

    # generate augmented images
    image_dict = {'s': image,  # the original image
                  'sc': contrast(image),  # adjust the contrast ratio
                  'sb': bright(image),  # adjust the brightness
                  's_f0': cv2.flip(image, 0),  # flip the image vertically
                  's_f0c': contrast(cv2.flip(image, 0)),  # flip & contrast
                  's_f0b': bright(cv2.flip(image, 0)),  # flip & bright
                  's_f1': cv2.flip(image, 1),  # flip the image horizontally
                  's_f1c': contrast(cv2.flip(image, 1)),  # flip & contrast
                  's_f1b': bright(cv2.flip(image, 1)),  # flip & bright
                  's_r_90': cv2.warpAffine(image, matRotate_90, (0, 0)),  # rotate -90 degrees
                  's_r_90c': contrast(cv2.warpAffine(image, matRotate_90, (0, 0))),  # rotate & contrast
                  's_r_90b':    bright(cv2.warpAffine(image, matRotate_90, (0, 0))),  # rotate & bright
                  's_r90':      cv2.warpAffine(image, matRotate90, (0, 0)),  # rotate 90 degrees
                  's_r90c':     contrast(cv2.warpAffine(image, matRotate90, (0, 0))),  # rotate & contrast
                  's_r90b':     bright(cv2.warpAffine(image, matRotate90, (0, 0))),  # rotate & bright
                  's_r180':     cv2.warpAffine(image, matRotate180, (0, 0)),  # rotate 180 degrees
                  's_r180c':    contrast(cv2.warpAffine(image, matRotate180, (0, 0))),  # rotate & contrast
                  's_r180b':    bright(cv2.warpAffine(image, matRotate180, (0, 0)))  # rotate & bright
                  }

    # save the augmented images
    for suffix, image in image_dict.items():
        dice = random.random()      # generate a random number between 0-1

        if dice < ratio:
            filename = train_root + name + suffix + '.jpg'
            cv2.imwrite(filename, image)
        else:
            filename = val_root + name + suffix + '.jpg'
            cv2.imwrite(filename, image)
            print(filename)


if __name__ == "__main__":
    src_root = "C:/Users/31270/Desktop/graduation_project/New_data/0_ori"
    train_root = "C:/Users/31270/Desktop/graduation_project/New_data/train"
    val_root = "C:/Users/31270/Desktop/graduation_project/New_data/val"

    # loop through the source files and then generate datasets
    for filename in os.listdir(src_root):
        # get source image
        print(src_root + '/' + filename)
        image = cv2.imread(src_root + '/' + filename)

        # detect the cutting edge, then x+w is the x position of the edge
        x, y, w, h = measure_object(image)
        print(x, y, w, h)



