from __future__ import print_function, division
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import time
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
import cv2
import os
from model import load_model

'''
该代码为检测用户输入照片的代码
为了测试模型的泛化能力，选取了新旧数据集中的随机照片进行检测
先将照片分割为256*256的小块，依次进行预测以及投票
'''

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0")

# load the model
model_angle = load_model('F:/graduation_project/new_models/test_angle_SWA.pth', 2, 'cuda')
model_quality = load_model('F:/graduation_project/models/resnet18_quality.pth', 3, 'cuda')
model_speed = load_model('F:/graduation_project/models/resnet18_speed.pkl', 5, 'cuda')
model_focus = load_model('F:/graduation_project/models/resnet18_focus.pkl', 5, 'cuda')
model_pressure = load_model('F:/graduation_project/models/resnet18_pressure.pth', 5, 'cuda')

'''
from Resnet_improve import resnext50_32x4d
device = torch.device("cuda:0")
model_angle = resnext50_32x4d()
inchannel = model_angle.fc.in_features
model_angle.fc = nn.Linear(inchannel, 2)
model_weight_path = 'F:/graduation_project/new_models/angle_model_2.pth'
model_angle.load_state_dict(torch.load(model_weight_path), False)
missing_keys, unexpected_keys = model_angle.load_state_dict(torch.load(model_weight_path), strict=False)
for params in model_angle.parameters():
    params.requires_grad = False
'''
'''
from ResNeSt.resnest import resnest50
model_angle = resnest50(pretrained=False)
inchannel = model_angle.fc.in_features
model_angle.fc = nn.Linear(inchannel, 2)
model_weight_path = 'F:/graduation_project/new_models/angle_model_3.pth'
model_angle.load_state_dict(torch.load(model_weight_path), False)
missing_keys, unexpected_keys = model_angle.load_state_dict(torch.load(model_weight_path), strict=False)
'''

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

angle_class_names = ['0', '90']
quality_class_names = ['1', '2', '3']
focus_class_names = ['-2', '-2.8', '-3.5', '-4.3', '-5']
pressure_class_names = ['7', '7.5', '8.5', '9.5', '10']
speed_quality_names = ['6', '7.5', '9', '10.5', '12']


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dst = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, binary = cv2.threshold(dst, 125, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 100
    sum_area = 0
    pos_x = 0
    pos_y = 0
    width = 0
    height = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        sum_area = sum_area + area
        if area < max_area:
            continue
        else:
            max_area = area
            pos_x, pos_y, width, height = cv2.boundingRect(contour)

    return pos_x, pos_y, width, height


def measure_object90(image):
    """
    this function is for 90° photos

    input:
        image: the photo of the workpiece
    output:
        (x_pos, y_pos):     the left-down point of the workpiece
        (width, height):    the size of the workpiece
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contour 内储存xy向量表示曲线，但是有可能为无面积

    max_area = 0
    x_pos = 0
    y_pos = 0
    width = 0
    height = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max_area:
            continue
        else:
            max_area = area
            x_pos, y_pos, width, height = cv2.boundingRect(contour)

    if x_pos > 1200 or x_pos < 600:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        binary = cv2.dilate(binary, kernel)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # contour 内储存xy向量表示曲线，但是有可能为无面积

        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max_area:
                continue
            else:
                max_area = area
                x_pos, y_pos, width, height = cv2.boundingRect(contour)

    return x_pos, y_pos, width, height


def measure_object0(image):
    """
    this function is for 0° photos

    input:
        image: the photo of the workpiece
    output:
        (x_pos, y_pos):     the left-down point of the workpiece
        (width, height):    the size of the workpiece
    """

    media = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(media, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

    cv2.imwrite("./img.jpg", image)
    cv2.waitKey(0)

    max_area = 0
    pos_x = 0
    pos_y = 0
    width = 0
    height = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < max_area:
            continue
        else:
            max_area = area
            pos_x, pos_y, width, height = cv2.boundingRect(contour)
    return pos_x, pos_y, width, height


def val_ori(image, angle, model, class_names, transforms):
    # pre-processing
    row, col = image.shape[:2]
    if angle == 90:
        x_ori, y_ori, w, h = measure_object90(image)
        # print(row, col, x_ori, y_ori, w, h)
        # plt.imshow(image[:, :, ::-1])
        # plt.plot([x_ori, x_ori + w, x_ori + w, x_ori, x_ori], [y_ori, y_ori, y_ori + h, y_ori + h, y_ori], linewidth=3)
        # plt.show()
        input_list = []
        y = y_ori
        for _ in range(int(h / 256)):
            x = x_ori
            for _ in range(int(w / 256)):
                tmp_image = image[y: y + 256, x: x + 256]
                tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
                PIL_image = Image.fromarray(tmp_image)
                input = transforms(PIL_image)
                input.unsqueeze_(0)
                input_list.append(input)
                x += 256
            y += 256
    elif angle == 0:
        x, y, w, h = measure_object0(image)
        input_list = []
        for y in range(0, row - 256, 256):
            input_image = image[y: y + 256, x + w - 100: x + w + 156]
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(input_image)
            input = transforms(PIL_image)
            input.unsqueeze_(0)
            input_list.append(input)
    # create list
    index_list = np.zeros(len(class_names))
    # vote
    for input in input_list:
        outputs = model(input)
        _, predict = torch.max(outputs, 1)
        index_list[int(predict)] += 1
    # find the most probable parameters
    index = 0
    for i in range(len(class_names)):
        if index_list[index] < index_list[i]:
            index = i
    return class_names[index]


def angle_detecting(image, model, class_names, transforms):
    """
    input:
        image:          the photo of the workpiece
        model:          the CNN model to detect the angle
        transforms:     the transformation before classification
    output:
        angle_index:      the predicted angle {0: 0, 1: 45, 2: 90}
    """

    # locate the edge of the workpiece
    row, col = image.shape[:2]
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_list = []
    for x in range(0, col - 1024, 512):
        for y in range(0, row - 1024, 512):
            candidate = image_gray[y: y+1024, x: x+1024]

            fft = fft2(candidate)
            fft_shift = fftshift(fft)

            direct_energy = abs(fft_shift[512][512])**2
            total_energy = 0
            for i in range(512, 520):
                for j in range(512, 520):
                    total_energy += abs(fft_shift[i][j])**2

            energy_rate = direct_energy / total_energy
            if energy_rate < 0.99:
                image_list.append(image[y: y+1024, x: x+1024])
                cv2.rectangle(image, (x, y), (x+1024, y+1024), (255, 10, 10), 10)

    # detect the angle
    angle_list = [0, 0]      # 0, 90
    for selected_image in image_list:
        # pre-processing
        selected_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(selected_image)
        test_image = transforms(PIL_image)
        test_image.unsqueeze_(0)

        # do the classification
        outputs = model(test_image)
        _, predict_angle = torch.max(outputs, 1)
        angle_list[int(predict_angle)] += 1

    # find the most probable angle
    angle_index = 0
    if angle_list[angle_index] < angle_list[1]:
        angle_index = 1

    return class_names[angle_index]


if __name__ == '__main__':
    # 照片集测试，展示准确率
    correct = 0
    total = 0
    path = 'F:/graduation_project/old_data/val_ori/angle'
    since = time.time()
    for target in angle_class_names:
        print("标签为", target)
        folder_path = path + '/' + target
        path_list = os.listdir(folder_path)
        for folder_name in path_list:
            workpiece_path = folder_path + '/' + folder_name
            pic_list = os.listdir(workpiece_path)
            for filename in pic_list:
                pic_path = workpiece_path + '/' + filename
                image = cv2.imread(pic_path)
                predict = angle_detecting(image, model_angle, angle_class_names, transforms)
                # predict = val_ori(image, 90, model_angle, angle_class_names, transforms)
                print("预测结果为", predict)

                if predict == target:
                    correct += 1
                total += 1
    time_elapsed = time.time() - since
    acc = correct / total * 100.0
    print("accuracy:{:.2f}%[{:d}/{:d}]".format(acc, correct, total))
    print("time:{:.0f}s".format(time_elapsed))

    '''
    # 单个图片测试
    image = cv2.imread("1_0.jpg")
    pressure = val_ori(image, 0, model_pressure, pressure_class_names, transforms)
    print("压力为", pressure)
    '''
