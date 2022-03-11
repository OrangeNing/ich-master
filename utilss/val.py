
import torch
from torch.backends import cudnn
import torch.nn as nn
from model import cnn_models
import os
import glob
import numpy as np
import pandas as pd
import torch.optim as optim
from utilss import test_csv_any
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import gc
import datetime
from utilss import test_csv
from utilss import loss
from albumentations import (Compose, Normalize, HorizontalFlip,
                            ShiftScaleRotate, Transpose
                            )
from apex import amp
import warnings
from utilss.logs import get_logger
from utilss.utils import dumpobj
from datasets import  cnn_datasets
def createIndex_sick(imgname,sick):
    index_list = []
    for name in imgname:
        # for type in label_cols[]:
        index_list.append(name+'_'+sick)
    return index_list
def mymodel(model):
    para_model = None
    if model == 'resnest101':
        para_model = cnn_models.resnest101()
    if model == 'resnext101':
        para_model = cnn_models.resnext101_32x8d()
    if model == 'se_resnext':
        para_model = cnn_models.se_resnext101_32x4d()
    return  para_model
path_data = '/media/ps/_data/ICH/rsna-master/data'
dir_train_img = os.path.join(path_data, 'dataset/train/')
dir_test_img = os.path.join(path_data, 'dataset/test/')
# Parameters
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
train = pd.read_csv(os.path.join(path_data, 'mytrain.csv'))
test = pd.read_csv(os.path.join(path_data, 'mytest.csv'))
# Data loaders
mean_img = [0.22363983, 0.18190407, 0.2523437]
std_img = [0.32451536, 0.2956294, 0.31335256]
transform_train = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                     rotate_limit=20, p=0.3, border_mode=cv2.BORDER_REPLICATE),
    Transpose(p=0.5),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])
HFLIPVAL = 1.0
TRANSPOSEVAL = 1.0
transform_test = Compose([
    HorizontalFlip(p=HFLIPVAL),
    Transpose(p=TRANSPOSEVAL),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])
batch_size = 40
SIZE = 448
AUTOCROP = True
trn_sick = label_cols[0]
device = 'cuda'
tstdataset = cnn_datasets.IntracranialDataset(test, path=dir_test_img, transform=transform_test, labels=False,pred = True,
                                              SIZE=SIZE, label_cols=label_cols, AUTOCROP=AUTOCROP, trn_sick=trn_sick)
num_workers = 16
tstloader = DataLoader(tstdataset, batch_size=batch_size * 8, shuffle=True, num_workers=num_workers)
model = mymodel('resnest101')
for param in model.parameters():
    param.requires_grad = False
input_model_file ='/media/ps/_data/ICH/rsna-master/scripts/resnest101vfusion/weights/448_epoch26_fold4_sickepidural.bin'
model.load_state_dict(torch.load(input_model_file),False)
model.to(device)
lr = 0.00005
plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")#01
n_gpu = torch.cuda.device_count()
model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
print(">>>>start val ")
pred_list = []
index_list = []
for step, batch in enumerate(tstloader):
    inputs = batch["image"]
    inputs = inputs.to(device, dtype=torch.float)
    logits = model(inputs)
    pred_out = torch.sigmoid(logits)
    pred_out = pred_out.cpu().numpy()
    pred_out = pred_out.ravel().tolist()
    index_out = createIndex_sick(batch['imgname'],trn_sick)
    pred_list.append(pred_out)
    index_list.append(index_out)
pred_list = np.concatenate(pred_list,0)
index_list = np.concatenate(index_list,0)
pred_dict = {'ID': index_list, 'Label': pred_list}
pred_dict_df = pd.DataFrame(pred_dict)
# index_list = np.concatenate(index_list,0).tolist()
# pred_dict = {'ID': index_list, 'Label': pred_list}
# gt_dict = {'ID': index_list, 'Label': gt_list}
# pred_dict_df = pd.DataFrame(pred_dict)
# gt_dict_df = pd.DataFrame(pred_dict)
# print('>>> epoch{}>>>{}>>>'.format(epoch,model_para))
lossfun = 'bce'
epoch_loss = 0
epoch=0
model_para='resnest101'
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# predict_csv,trn_sick,lossfun,epoch_loss,epoch,model_para,time
test_csv_any.test_sick_any(pred_dict_df,0,lossfun,epoch_loss,epoch,model_para,time)
print("end val!")