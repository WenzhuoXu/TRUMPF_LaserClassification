import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from qualityTrain import LaserCutEvalDataset, Attributes
from qualityTrain import NeuralNetwork
from qualityTrain import calculate_metrics

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

testing_index = 'F:/BachelorThesis/Data/data2021_ori/90_ori_testing_index.txt'
img_dir = 'F:/BachelorThesis/Data/data_highfreq'


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_speed_all = []
    gt_focus_all = []
    gt_pressure_all = []
    gt_quality_all = []
    predicted_speed_all = []
    predicted_focus_all = []
    predicted_pressure_all = []
    predicted_quality_all = []

    accuracy_speed = 0
    accuracy_focus = 0
    accuracy_pressure = 0
    accuracy_quality = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['image']
            gt_speed = batch['labels']['speed']
            gt_focus = batch['labels']['focus']
            gt_pressure = batch['labels']['pressure']
            gt_quality = batch['labels']['quality']
            output = model(img.to(device))

            batch_accuracy_speed, batch_accuracy_focus, batch_accuracy_pressure, batch_accuracy_quality = \
                calculate_metrics(output, batch['labels'])
            accuracy_speed += batch_accuracy_speed
            accuracy_focus += batch_accuracy_focus
            accuracy_pressure += batch_accuracy_pressure
            accuracy_quality += batch_accuracy_quality

            # get the most confident prediction for each image
            _, predicted_speed = output['speed'].cpu().max(1)
            _, predicted_focus = output['focus'].cpu().max(1)
            _, predicted_pressure = output['pressure'].cpu().max(1)
            _, predicted_quality = output['quality'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_speed = attributes.speed_id_to_name[predicted_speed[i].item()]
                predicted_focus = attributes.focus_id_to_name[predicted_focus[i].item()]
                predicted_pressure = attributes.pressure_id_to_name[predicted_pressure[i].item()]
                predicted_quality = attributes.quality_id_to_name[predicted_quality[i].item()]

                gt_speed = attributes.speed_id_to_name[gt_speed[i].item()]
                gt_focus = attributes.focus_id_to_name[gt_focus[i].item()]
                gt_pressure = attributes.pressure_id_to_name[gt_pressure[i].item()]
                gt_quality = attributes.quality_id_to_name[gt_quality[i].item()]

                gt_speed_all.append(gt_speed)
                gt_focus_all.append(gt_focus)
                gt_pressure_all.append(gt_pressure)

                predicted_speed_all.append(predicted_speed)
                predicted_focus_all.append(predicted_focus)
                predicted_pressure_all.append(predicted_pressure)
                predicted_quality_all.append(predicted_quality)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_speed, predicted_focus, predicted_pressure,
                                                  predicted_quality))
                gt_labels.append("{}\n{}\n{}".format(gt_speed, gt_focus, gt_pressure, gt_quality))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nspeed: {:.4f}, focus: {:.4f}, pressure: {:.4f}, quality: {:.4f}".format(
            accuracy_speed / n_samples,
            accuracy_focus / n_samples,
            accuracy_pressure / n_samples,
            accuracy_quality / n_samples
        ))

        # 绘制混淆矩阵
        if show_cn_matrices:
            # color
            cn_matrix = confusion_matrix(
                y_true=gt_speed_all,
                y_pred=predicted_speed_all,
                labels=attributes.speed_labels,
                normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.speed_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Speed")
        plt.tight_layout()
        plt.show()

        # gender
        cn_matrix = confusion_matrix(
            y_true=gt_focus_all,
            y_pred=predicted_focus_all,
            labels=attributes.focus_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.focus_labels).plot(
            xticks_rotation='horizontal')
        plt.title("Focus")
        plt.tight_layout()
        plt.show()

        # 取消下面代码的注释，查看物品混淆矩阵（可能太大无法显示）
        cn_matrix = confusion_matrix(
            y_true=gt_pressure_all,
            y_pred=predicted_pressure_all,
            labels=attributes.pressure_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.pressure_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Pressure")
        plt.tight_layout()
        plt.show()

        cn_matrix = confusion_matrix(
            y_true=gt_quality_all,
            y_pred=predicted_quality_all,
            labels=attributes.quality_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.quality_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Quality")
        plt.tight_layout()
        plt.show()

        if show_images:
            labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 2
        n_rows = 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, default=r'checkpoints\2022-02-25_17-13\checkpoint-001000.pth',
                        help="Path to the checkpoint")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # 属性变量包含数据集中类别的标签以及字符串名称和 ID 之间的映射
    attributes = Attributes()

    # 在验证期间，我们只使用张量和归一化变换
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = LaserCutEvalDataset(testing_index, img_dir, test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = NeuralNetwork(n_speed_classes=attributes.num_speed, n_focus_classes=attributes.num_focus,
                          n_pressure_classes=attributes.num_pressure, n_quality_classes=attributes.num_quality).to(
        device)

    # 训练模型的可视化
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)