import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from qualityTrainV2 import LaserCutEvalDataset, Attributes
from qualityTrainV2 import NeuralNetwork
from qualityTrainV2 import calculate_metrics

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

training_index = 'Data/data2021_ori/90_ori_training_index.txt'
testing_index = 'Data/data2021_ori/90_ori_testing_index.txt'
img_dir = 'Data/data_highfreq'


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def visualize_grid(model, dataloader, attributes, device, checkpoint=None, show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    speed_labels = [6.0, 7.5, 9.0, 10.5, 12]
    focus_labels = [-2.0, -2.8, -3.5, -4.3, -5.0]
    imgs = []
    labels = []
    gt_labels = []
    gt_speed_all = []
    gt_focus_all = []
    # gt_pressure_all = []
    gt_quality_all = []
    predicted_speed_all = []
    predicted_focus_all = []
    # predicted_pressure_all = []
    predicted_quality_all = []

    accuracy_speed = 0
    accuracy_focus = 0
    # accuracy_pressure = 0
    accuracy_quality = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['image']
            gt_speed = batch['labels']['speed']
            gt_focus = batch['labels']['focus']
            # gt_pressure = batch['labels']['pressure']
            gt_quality = batch['labels']['quality']
            output = model(img.to(device))

            batch_accuracy_speed, batch_accuracy_focus, batch_accuracy_quality = \
                calculate_metrics(output, batch['labels'])
            accuracy_speed += batch_accuracy_speed
            accuracy_focus += batch_accuracy_focus
            # accuracy_pressure += batch_accuracy_pressure
            accuracy_quality += batch_accuracy_quality

            # get the most confident prediction for each image
            predicted_speed = output['speed'].cpu().float()
            predicted_focus = output['focus'].cpu().float()
            # _, predicted_pressure = output['pressure'].cpu().max(1)
            _, predicted_quality = output['quality'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                # predicted_speed[i] = attributes.speed_id_to_name[predicted_speed[i].item()]
                # predicted_focus[i] = attributes.focus_id_to_name[predicted_focus[i].item()]
                # predicted_pressure[i] = attributes.pressure_id_to_name[predicted_pressure[i].item()]
                predicted_quality[i] = attributes.quality_id_to_name[predicted_quality[i].item()]

                # gt_speed[i] = attributes.speed_id_to_name[gt_speed[i].item()]
                # gt_focus[i] = attributes.focus_id_to_name[gt_focus[i].item()]
                # gt_pressure[i] = attributes.pressure_id_to_name[gt_pressure[i].item()]
                gt_quality[i] = attributes.quality_id_to_name[gt_quality[i].item()]

                gt_speed_all.append(gt_speed[i].item())
                gt_focus_all.append(gt_focus[i].item())
                # gt_pressure_all.append(gt_pressure[i])
                gt_quality_all.append(gt_quality[i])

                predicted_speed_all.append(predicted_speed[i].item())
                predicted_focus_all.append(predicted_focus[i].item())
                # predicted_pressure_all.append(predicted_pressure[i])
                predicted_quality_all.append(predicted_quality[i])

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_speed[i], predicted_focus[i], predicted_quality[i]))
                gt_labels.append("{}\n{}\n{}".format(gt_speed[i], gt_focus[i], gt_quality[i]))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nspeed: {:.4f}, focus: {:.4f}, quality: {:.4f}".format(
            accuracy_speed / n_samples,
            accuracy_focus / n_samples,
            accuracy_quality / n_samples
        ))
        '''
        # 绘制混淆矩阵
        if show_cn_matrices:
            # color
            cn_matrix = confusion_matrix(
                y_true=gt_speed_all,
                y_pred=predicted_speed_all,
                labels=[6, 7, 9, 10, 12],
                normalize='true')
        ConfusionMatrixDisplay(cn_matrix).plot(
           include_values=True, xticks_rotation='vertical')
        plt.title("Speed")
        plt.tight_layout()
        plt.show()

        # gender
        cn_matrix = confusion_matrix(
            y_true=gt_focus_all,
            y_pred=predicted_focus_all,
            labels=[-2, -3, -4, -5],
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix).plot(
            include_values=True, xticks_rotation='horizontal')
        plt.title("Focus")
        plt.tight_layout()
        plt.show()
        '''
        speed6 = []
        speed75 = []
        speed9 = []
        speed105 = []
        speed12 = []
        for i in range(1, len(predicted_focus_all), 1):
            if gt_speed_all[i] == 6.0:
                speed6.append(predicted_speed_all[i])

            if gt_speed_all[i] == 7.5:
                speed75.append(predicted_speed_all[i])

            if gt_speed_all[i] == 9.0:
                speed9.append(predicted_speed_all[i])

            if gt_speed_all[i] == 10.5:
                speed105.append(predicted_speed_all[i])

            if gt_speed_all[i] == 12:
                speed12.append(predicted_speed_all[i])
        speed_avg = [np.mean(speed6), np.mean(speed75), np.mean(speed9), np.mean(speed105), np.mean(speed12)]
        speed_arr = [np.std(speed6), np.std(speed75), np.std(speed9), np.std(speed105), np.std(speed12)]

        focus2 = []
        focus28 = []
        focus35 = []
        focus43 = []
        focus5 = []

        for i in range(1, len(predicted_focus_all), 1):
            if gt_focus_all[i] == -2.0:
                focus2.append(predicted_focus_all[i])

            if gt_focus_all[i] == -2.8:
                focus28.append(predicted_focus_all[i])

            if gt_focus_all[i] == -3.5:
                focus35.append(predicted_focus_all[i])

            if gt_focus_all[i] == -4.3:
                focus43.append(predicted_focus_all[i])

            if gt_focus_all[i] == -5.0:
                focus5.append(predicted_focus_all[i])
        focus_avg = [np.mean(focus2), np.mean(focus28), np.mean(focus35), np.mean(focus43), np.mean(focus5)]
        focus_arr = [np.std(focus2), np.std(focus28), np.std(focus35), np.std(focus43), np.std(focus5)]

        plt.figure()
        plt.errorbar(speed_labels, speed_avg, yerr=speed_arr, fmt='bo:')
        plt.xlabel('ground truth speed')
        plt.ylabel('predicted speed')
        plt.show()

        plt.figure()
        plt.errorbar(focus_labels, focus_avg, yerr=focus_arr, fmt='bo:')
        plt.xlabel('ground truth focus')
        plt.ylabel('predicted focus')
        plt.show()

        cn_matrix = confusion_matrix(
            y_true=gt_quality_all,
            y_pred=predicted_quality_all,
            labels=[1, 2, 3],
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix).plot(
            include_values=True, xticks_rotation='vertical')
        plt.figure()
        plt.title("Quality")
        plt.tight_layout()
        plt.show()
        '''
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
        '''
        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, default=r'checkpoints\2022-03-17_15-06\checkpoint-000700.pth',
                        help="Path to the checkpoint")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # device = torch.device("cpu")

    # 属性变量包含数据集中类别的标签以及字符串名称和 ID 之间的映射
    attributes = Attributes(training_index)

    # 在验证期间，我们只使用张量和归一化变换
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = LaserCutEvalDataset(testing_index, img_dir, test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = NeuralNetwork(n_quality_classes=attributes.num_quality).to(device)

    # 训练模型的可视化
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)
