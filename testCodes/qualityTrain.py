import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

training_index = 'F:/Work/Bachelor-thesis/Data/data2021_ori/90_ori_training_index.txt'
testing_index = 'F:/Work/Bachelor-thesis/Data/data2021_ori/90_ori_testing_index.txt'
img_dir = 'F:/Work/Bachelor-thesis/Data/data_highfreq'


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
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
