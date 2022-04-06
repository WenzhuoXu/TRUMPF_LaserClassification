import os
import warnings
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, r2_score, explained_variance_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms
from torchvision.io import read_image

training_index = 'Data/training_index.txt'
testing_index = 'Data/testing_index.txt'
img_dir = 'Data/4mm_processed'


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        # accuracy_speed = 0
        accuracy_focus = 0
        # accuracy_pressure = 0
        # accuracy_quality = 0

        for batch in dataloader:
            img = batch['image']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_focus = \
                calculate_metrics(output, target_labels)

            # accuracy_speed += batch_accuracy_speed
            accuracy_focus += batch_accuracy_focus
            # accuracy_pressure += batch_accuracy_pressure
            # accuracy_quality += batch_accuracy_quality

    n_samples = len(dataloader)
    avg_loss /= n_samples
    # accuracy_speed /= n_samples
    accuracy_focus /= n_samples
    # accuracy_pressure /= n_samples
    # accuracy_quality /= n_samples
    print('-' * 72)
    print("Validation  loss: {:.4f}, focus: {:.4f}\n".format(
        avg_loss, accuracy_focus))

    logger.add_scalar('val_loss', avg_loss, iteration)
    # logger.add_scalar('val_accuracy_speed', accuracy_speed, iteration)
    logger.add_scalar('val_accuracy_focus', accuracy_focus, iteration)
    # logger.add_scalar('val_accuracy_pressure', accuracy_pressure, iteration)
    # logger.add_scalar('val_accuracy_quality', accuracy_quality, iteration)

    model.train()


class Attributes:
    def __init__(self, annotation_file):
        # self.speed_labels = [6, 7.5, 9, 10.5, 12]
        # self.focus_labels = [-2, -2.8, -3.5, -4.3, -5]
        # self.pressure_labels = [7, 7.8, 8.5, 9.3, 10]
        self.quality_labels = [1, 2, 3, 4, 5]

        # self.num_speed = len(self.speed_labels)
        # self.num_focus = len(self.focus_labels)
        # self.num_pressure = len(self.pressure_labels)
        self.num_quality = len(self.quality_labels)

        # self.speed_id_to_name = dict(zip(range(len(self.speed_labels)), self.speed_labels))
        # self.speed_name_to_id = dict(zip(self.speed_labels, range(len(self.speed_labels))))

        # self.focus_id_to_name = dict(zip(range(len(self.focus_labels)), self.focus_labels))
        # self.focus_name_to_id = dict(zip(self.focus_labels, range(len(self.focus_labels))))

        # self.pressure_id_to_name = dict(zip(range(len(self.pressure_labels)), self.pressure_labels))
        # self.pressure_name_to_id = dict(zip(self.pressure_labels, range(len(self.pressure_labels))))

        self.quality_id_to_name = dict(zip(range(len(self.quality_labels)), self.quality_labels))
        self.quality_name_to_id = dict(zip(self.quality_labels, range(len(self.quality_labels))))


class LaserCutEvalDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_data = pd.read_table(annotations_file, sep='\t', header=None, names=None)
        self.img_content = self.img_data[0]
        # self.speed = self.img_data[1]
        self.focus = self.img_data[2]
        # self.pressure = self.img_data[3]
        # self.quality = self.img_data[3]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_content)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_content.iloc[idx])
        image = read_image(img_path)

        # speed = self.speed.iloc[idx]
        focus = self.focus.iloc[idx]
        # pressure = self.pressure.iloc[idx]
        # quality = self.quality.iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)

        dict_data = {
            'image': image,
            'labels': {
                # 'speed': speed,
                'focus': focus,
                # 'pressure': pressure,
                # 'quality': quality,
            }
        }
        return dict_data


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                            shear=None, resample=False, fillcolor=(255, 255, 255)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_data = LaserCutEvalDataset(training_index, img_dir, train_transforms)
testing_data = LaserCutEvalDataset(testing_index, img_dir, test_transforms)

training_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, drop_last=True)
testing_dataloader = DataLoader(testing_data, batch_size=16, shuffle=True, drop_last=True)

n_train_samples = len(training_dataloader)

# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class NeuralNetwork(nn.Module):
    def __init__(self, n_quality_classes):
        super(NeuralNetwork, self).__init__()
        self.base_model = models.mobilenet_v3_large(pretrained=True).features
        # self.base_model = models.mobilenet_v2(pretrained='True').features
        last_channel = models.mobilenet_v3_large(pretrained=True).lastconv_output_channels

        self.pool = nn.AdaptiveAvgPool2d(1)

        # self.speed = nn.Sequential(
        #     nn.Linear(in_features=last_channel, out_features=1000),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(in_features=1000, out_features=1),
        # )
        self.focus = nn.Sequential(
            nn.Linear(in_features=last_channel, out_features=1000),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1000, out_features=1)
        )
        # self.pressure = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=last_channel, out_features=1)
        # )
        # self.quality = nn.Sequential(
        #     nn.Linear(in_features=last_channel, out_features=1000),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(in_features=1000, out_features=n_quality_classes)
        # )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        return {
            # 'speed': self.speed(x).squeeze(-1),
            'focus': self.focus(x).squeeze(-1),
            # 'pressure': self.pressure(x),
            # 'quality': self.quality(x)
        }

    def get_loss(self, net_output, ground_truth):
        loss_fn_p = nn.SmoothL1Loss(reduction='mean')
        # speed_loss = loss_fn_p(net_output['speed'].float(), ground_truth['speed'].float())
        focus_loss = loss_fn_p(net_output['focus'].float(), ground_truth['focus'].float())
        # pressure_loss = F.cross_entropy(net_output['pressure'], ground_truth['pressure'])
        # quality_loss = F.cross_entropy(net_output['quality'], ground_truth['quality'])

        loss = focus_loss
        return loss, {'focus': focus_loss}


def calculate_metrics(output, target):
    # predicted_speed = output['speed'].cpu().float()
    # gt_speed = target['speed'].cpu()

    predicted_focus = output['focus'].cpu().float()
    gt_focus = target['focus'].cpu()

    # _, predicted_pressure = output['pressure'].cpu()
    # gt_pressure = target['pressure'].cpu()

    # _, predicted_quality = output['quality'].cpu().max(1)
    # gt_quality = target['quality'].cpu()

    with warnings.catch_warnings():  # sklearn 在处理混淆矩阵中的零行时可能会产生警告
        warnings.simplefilter("ignore")
        # accuracy_speed = balanced_accuracy_score(y_true=gt_speed.numpy(), y_pred=predicted_speed.numpy())
        # accuracy_speed = r2_score(gt_speed.detach().numpy(), predicted_speed.detach().numpy(), multioutput="uniform_average")
        # accuracy_focus = balanced_accuracy_score(y_true=gt_focus.numpy(), y_pred=predicted_focus.numpy())
        accuracy_focus = r2_score(gt_focus.detach().numpy(), predicted_focus.detach().numpy(), multioutput="uniform_average")
        # accuracy_pressure = balanced_accuracy_score(y_true=gt_pressure.numpy(), y_pred=predicted_pressure.numpy())
        # accuracy_pressure = abs(abs(predicted_pressure - gt_pressure) / gt_pressure)
        # accuracy_quality = balanced_accuracy_score(y_true=gt_quality.numpy(), y_pred=predicted_quality.numpy())

    return accuracy_focus


if __name__ == '__main__':
    logdir = os.path.join('./logs/', get_cur_time())
    savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    start_epoch = 1
    N_epochs = 1000
    # batch_size = 16
    # num_workers = 6

    attributes = Attributes(training_index)

    model = NeuralNetwork(n_quality_classes=attributes.num_quality).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    print('Start training...')

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        # accuracy_speed = 0
        accuracy_focus = 0
        # accuracy_pressure = 0
        # accuracy_quality = 0
        for batch in training_dataloader:
            optimizer.zero_grad()
            img = batch['image']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_focus = \
                calculate_metrics(output, target_labels)
            # accuracy_speed += batch_accuracy_speed
            accuracy_focus += batch_accuracy_focus
            # accuracy_pressure += batch_accuracy_pressure
            # accuracy_quality += batch_accuracy_quality

            loss_train.backward()
            optimizer.step()
        print("prediction focus: {:.4f}".format(
            # output['speed'][1],
            output['focus'][1],)
        )

        print("target focus: {:.4f}".format(
            # target_labels['speed'][1],
            target_labels['focus'][1],)
        )

        print("epoch {:4d}, loss: {:.4f}, focus: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            # accuracy_speed / n_train_samples,
            accuracy_focus / n_train_samples
            # accuracy_quality / n_train_samples, )
        ))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)
        # logger.add_scalar('train_accu_speed', accuracy_speed / n_train_samples, epoch)
        logger.add_scalar('train_accu_focus', accuracy_focus / n_train_samples, epoch)
        # logger.add_scalar('train_accu_quality', accuracy_quality / n_train_samples, epoch)

        if epoch % 5 == 0:
            validate(model, testing_dataloader, logger, epoch, device)

        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)
