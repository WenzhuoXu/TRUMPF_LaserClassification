import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.io import read_image

training_index = 'F:/Work/Bachelor-thesis/Data/data2021_ori/90_ori_training_index.txt'
testing_index = 'F:/Work/Bachelor-thesis/Data/data2021_ori/90_ori_testing_index.txt'
img_dir = 'F:/Work/Bachelor-thesis/Data/data_highfreq'


class attributes:
    n_speed: [6, 7.5, 9, 10.5, 12]
    n_focus: [-2, -2.8, -3.5, -4.3, -5]
    n_pressure: [7, 7.8, 8.5, 9.3, 10]
    n_quality: [1, 2, 3]


class LaserCutEvalDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_data = pd.read_table(annotations_file, sep='\t')
        self.img_content = self.img_data[0]
        self.speed = self.img_data[1]
        self.focus = self.img_data[2]
        self.pressure = self.img_data[3]
        self.quality = self.img_data[4]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_content)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_content.iloc[idx, 0])
        image = read_image(img_path)

        speed = self.speed.iloc[idx]
        focus = self.focus.iloc[idx]
        pressure = self.pressure.iloc[idx]
        quality = self.quality.iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)

        dict_data = {
            'image': image,
            'labels': {
                'speed': speed,
                'focus': focus,
                'pressure': pressure,
                'quality': quality,
            }
        }
        return dict_data


train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                            shear=None, resample=False, fillcolor=(255, 255, 255)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
])

training_data = LaserCutEvalDataset(training_index, img_dir, train_transforms)
testing_data = LaserCutEvalDataset(testing_index, img_dir, test_transforms)

training_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=16, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class NeuralNetwork(nn.Module):
    def __init__(self, n_speed_classes, n_focus_classes, n_pressure_classes, n_quality_classes):
        super(NeuralNetwork, self).__init__()

        self.base_model = models.mobilenet_v2().features
        last_channel = models.mobilenet_v2().last_channel

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.speed = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_speed_classes)
        )
        self.focus = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_focus_classes)
        )
        self.pressure = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_pressure_classes)
        )
        self.quality = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_quality_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        return {
            'speed': self.speed(x),
            'focus': self.focus(x),
            'pressure': self.pressure(x),
            'quality': self.quality(x)
        }

    def get_loss(self, net_output, ground_truth):
        speed_loss = F.cross_entropy(net_output['speed', ground_truth['speed']])
        focus_loss = F.cross_entropy(net_output['focus', ground_truth['focus']])
        pressure_loss = F.cross_entropy(net_output['pressure', ground_truth['pressure']])
        quality_loss = F.cross_entropy(net_output['quality', ground_truth['quality']])

        loss = speed_loss + focus_loss + pressure_loss + quality_loss
        return loss, {'speed': speed_loss, 'focus': focus_loss, 'pressure': pressure_loss, 'quality': quality_loss}


start_epoch = 1
N_epochs = 50
batch_size = 16
num_workers = 6

model = NeuralNetwork(n_speed_classes=attributes.n_speed, n_focus_classes=attributes.n_focus,
                      n_pressure_classes=attributes.n_pressure, n_quality_classes=attributes.n_quality).to(device)

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(start_epoch, N_epochs + 1):
    for batch in training_dataloader:
        optimizer.zero_grad()
