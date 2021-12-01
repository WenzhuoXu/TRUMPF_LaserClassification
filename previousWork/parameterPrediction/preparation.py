!pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 torchaudio===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
!pip3 install tqdm

import zipfile, os
filename = "/content/focus_train.zip"
zip_file = zipfile.ZipFile(filename)
os.mkdir(filename + '_files')
for folder in zip_file.namelist():
    zip_file.extract(folder, filename + '_files')
print("done")


# get list of models
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
# load pretrained models, using ResNeSt-50 as an example
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
for params in model.parameters():
    params.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0")
model = model.to(device)

!pip3 install torchcontrib