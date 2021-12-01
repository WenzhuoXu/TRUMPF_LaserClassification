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
model_weight_path = 'F:/graduation_project/new_models/quality_ResNeXt50.pth'
model_quality = load_model("resnext50", model_weight_path, 3)
model_pressure = load_model("resnext50", 'F:/graduation_project/new_models/pressure_resnext.pth', 5)
model_speed = load_model("resnet18", 'F:/graduation_project/new_models/test_speed_new.pth', 5)
model_focus = load_model("resnext50", 'F:/graduation_project/new_models/focus_resnext.pth', 5)

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

angle_class_names = ['0', '90']
quality_class_names = ['1', '2', '3']
focus_class_names = ['-2', '-2.8', '-3.5', '-4.3', '-5']
pressure_class_names = ['10', '7', '7.8', '8.5', '9.3']
speed_class_names = ['10.5', '12', '6', '7.5', '9']


if __name__ == '__main__':
    # 照片集测试，展示准确率
    correct = 0
    total = 0
    path = 'F:/graduation_project/val_ori/test/90/speed'
    since = time.time()

    target_number = 5
    model = model_speed
    class_names = speed_class_names
    for target in class_names:
        # 每个标签
        print("\n标签为{0}\n".format(target))
        folder_path = path + '/' + target
        workpiece_list = os.listdir(folder_path)
        for workpiece in workpiece_list:
            # 每个工件
            workpiece_path = folder_path + '/' + workpiece
            pic_list = os.listdir(workpiece_path)

            for picture in pic_list:
                # 每张图片
                pic_path = workpiece_path + '/' + picture
                block_list = os.listdir(pic_path)
                target_list = np.zeros(target_number)
                for block in block_list:
                    image_path = pic_path + '/' + block
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    PIL_image = Image.fromarray(image)
                    test_image = transforms(PIL_image)
                    test_image.unsqueeze_(0)
                    test_image = test_image.to(device)
                    # do the classification
                    outputs = model(test_image)
                    _, predict = torch.max(outputs, 1)
                    target_list[int(predict)] += 1
                index = 0
                for i in range(len(target_list)):
                    if target_list[i] > target_list[index]:
                        index = i

                print("{0}预测值为{1}".format(picture, class_names[index]))
                if float(class_names[index]) == float(target):
                    correct += 1
                total += 1

    time_elapsed = time.time() - since
    acc = correct / total * 100.0
    print("accuracy:{:.2f}%[{:d}/{:d}]".format(acc, correct, total))
    print("time:{:.0f}s".format(time_elapsed))
