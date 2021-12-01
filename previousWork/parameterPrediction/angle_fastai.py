import torch
from torch import nn
from torchvision import *
from fastai import *
from fastai.vision import *
from fastai.callbacks import CSVLogger, SaveModelCallback

# 创建learn
path = '/content/angle_mix.zip_files/train'
databunch = ImageDataBunch.from_folder(path, valid_pct=0.1)
learn = cnn_learner(databunch, models.resnet18, pretrained=True, metrics=[accuracy])
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(cyc_len=20)
learn.save('model_fastai.pth')
