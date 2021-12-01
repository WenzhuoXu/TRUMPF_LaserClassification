#!pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 torchaudio===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

import zipfile, os
filename = "/content/focus_train.zip"
zip_file = zipfile.ZipFile(filename)
os.mkdir(filename + '_files')
for folder in zip_file.namelist():
    zip_file.extract(folder, filename + '_files')
print("done")