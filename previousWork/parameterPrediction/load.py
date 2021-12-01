import torch
import torch.nn as nn
from Resnet_improve import resnext50_32x4d
from torchvision import models


def load_model(model, weight_path, fc_number, device_name="cuda:0"):
    device = torch.device("cuda:0")
    if model == "resnet18":
        net = models.resnet18()
    elif model == "resnext50":
        net = resnext50_32x4d()
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, fc_number)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.to(device)
    net.eval()
    return net