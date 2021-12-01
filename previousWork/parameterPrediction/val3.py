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
from load import load_model
from cut_0 import get_cut_image0
from xlwt import *

'''
该代码为检测用户输入照片的代码
为了测试模型的泛化能力，选取了新旧数据集中的随机照片进行检测
先将照片分割为256*256的小块，依次进行预测以及投票
'''

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0")

# load the model
'''
model_weight_path = 'F:/graduation_project/new_models/resnext50_quality.pth'
model_quality = load_model("resnet50", model_weight_path, 3)
'''
model_weight_path = 'F:/graduation_project/new_models/quality_mix.pth'
model_quality = load_model("resnet18", model_weight_path, 3)

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


quality_class_names = ['poor', 'acceptable', 'good']


if __name__ == '__main__':
    # 照片集测试，展示准确率
    correct = 0
    total = 0
    path = 'F:/graduation_project/3th/0'
    model = model_quality
    class_names = quality_class_names
    since = time.time()

    file = Workbook(encoding='utf-8')
    table = file.add_sheet('data')
    table.write(0, 0, "质量检测结果表")
    i = 1  # 当前行数
    workpiece_list = os.listdir(path)
    for workpiece in workpiece_list:
        workpiece_path = path + '/' + workpiece
        table.write(i, 0, "工件" + workpiece)
        i += 1
        pic_list = os.listdir(workpiece_path)
        for pic in pic_list:
            table.write(i, 0, "照片" + pic)
            pic_path = workpiece_path + '/' + pic
            image1 = cv2.imread(pic_path)
            rows, cols = image1.shape[:2]
            plt.imshow(image1[:, :, ::-1])
            plt.show()
            mid = int(input("请输入中间线的横坐标:"))
            mid = cols - mid
            image_list = get_cut_image0(image1, mid)
            quality_list = [0, 0, 0]
            for image in image_list:
                # 每个小块
                color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                PIL_image = Image.fromarray(color_image)
                test_image = transforms(PIL_image)
                test_image.unsqueeze_(0)
                test_image = test_image.to(device)
                # do the classification
                outputs = model(test_image)
                _, predict = torch.max(outputs, 1)
                quality_list[int(predict)] += 1
            print("Predicting Outcomes:", quality_list)
            # find biggest target
            quality_index = 0
            for index in range(3):
                if quality_list[quality_index] < quality_list[index]:
                    quality_index = index
            table.write(i, 1, class_names[quality_index])
            i += 1
    file.save('quality_val.xlsx')
    time_elapsed = time.time() - since
    print("time:{:.0f}s".format(time_elapsed))

