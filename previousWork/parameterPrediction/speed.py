from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
batch_size = 64

# set the dataloader
train_root = '/content/speed_new.zip_files/train'
val_root = '/content/speed_new.zip_files/val'

traindata_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valdata_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['10', '11', '12', '6', '8']

train_datasets = datasets.ImageFolder(train_root, traindata_transforms)
val_datasets = datasets.ImageFolder(val_root, valdata_transforms)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=64, shuffle=False, num_workers=8)

print(train_datasets.classes)
print(train_datasets.class_to_idx)
print(val_datasets.class_to_idx)
print("训练集照片数: ", len(train_datasets))
print("测试集照片数: ", len(val_datasets))

# instantiate the ResNet18 model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
device = torch.device("cuda:0")
model = model.to(device)


# train the model
def train(EPOCH, model, loader, device, optimizer, criterion):
    """
    train the model for an epoch on the train_datasets

    Args:
        model:      the model to be trained
        loader:     the data loader to feed training data to the model
        device:     the device on which the model will be trained
        optimizer:  the optimization algorithm to update the parameters in the model
        criterion:  the loss function to compute the loss between target classes and predicted classes

    Returns:
        model:      the trained model
    """

    model.train()
    running_loss = 0.0
    train_bar = tqdm(loader)
    for step, (data, targets) in enumerate(train_bar):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, EPOCH, loss)
    batch = len(train_datasets) // batch_size + 1
    print('train loss:{:.4f}'.format(running_loss/batch))
    return model, running_loss/batch


def val(model, loader, device, data_size, criterion):
    """
    test the model on val_datasets

    Args:
        model:      the model to be tested
        loader:     the data loader to feed test data to the model
        device:     the device on which the model will be tested
        data_size:  the number of images in test datasets
        criterion:  the loss function to compute the loss between target classes and predicted classes

    Returns:
        accuracy:   the accuracy of the model on the test datasets
    """

    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            test_loss += loss.data

            correct += (preds == targets.data).sum().item()

        test_loss = test_loss / len(loader)
        accuracy = correct / data_size * 100.0
        print('val loss:{:.4f},accuracy:{:.2f}% [{:d}/{:d}]'.format(test_loss, accuracy, correct, data_size))

        return accuracy, test_loss.item()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

since = time.time()
epoch_time = since
best_acc = 0
EPOCH = 30
Accuracy_list = []
train_loss_list = []
val_loss_list = []
for epoch in range(EPOCH):

    model, train_loss = train(EPOCH, model, train_loader, device, optimizer, criterion)
    torch.save(model.state_dict(), '/content/test_speed.pth')
    acc, val_loss = val(model, val_loader, device, len(val_datasets), criterion)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    if acc > best_acc:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = acc

    Accuracy_list.append(acc)
    train_time = time.time() - epoch_time
    epoch_time = time.time()
    print("Epoch {:d} complete in {:.0f}s".format(epoch + 1, train_time))
    exp_lr_scheduler.step()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

epoch = np.arange(1, EPOCH + 1, 1)
print("Accuracy_list =", Accuracy_list)
print("train_loss =", train_loss_list)
print("val_loss =", val_loss_list)
print("epoch =", epoch)

# save the model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), '/content/drive/MyDrive/test_speed_best.pth')
