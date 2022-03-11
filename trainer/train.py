import os
from configs import config_cnn as cfg
# import configs.config_cnn as cfg
from trainer import tools
import torch
from torch.backends import cudnn
import torch.nn as nn

import glob
import numpy as np
import pandas as pd
import torch.optim as optim
from apex import amp
import ast
from utilss import test_csv_any
from torch.utils.data import DataLoader
import datetime
from model import cnn_models, pvt, pvt_v2, repvgg
from utilss import loss
from model import cnn_models
import warnings
from datasets import cnn_datasets
import sys
import wandb
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest101

import torchvision
from model import resnet

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def wandbconfig(config, cfg):
    config.seed = cfg.seed
    config.fold = cfg.fold
    config.epochs = cfg.epochs
    config.batchsize = cfg.batchsize
    config.lr = cfg.lr
    config.size = cfg.size
    config.hflip = cfg.hflip  # Augmentation - Embedding horizontal flip
    config.transpose = cfg.transpose
    config.autocrop = cfg.autocrop
    config.loss = cfg.loss
    config.model = cfg.model
    config.weight_dacay = cfg.weight_dacay
    config.lr_gamma = cfg.lr_gamma
    config.lr_epoch = cfg.lr_epoch
    config.model1 = cfg.model1
    config.model2 = cfg.model2
    return config


# def data_distribution(train):
#     train_healthy = train[train['any'] == 1]
#     train_epidural = train[train['epidural']==1]
#     train_intraparenchymal = train[train['intraparenchymal']== 1]
#     train_intraventricular = train[train['intraventricular']==1]
#     train_subarachnoid = train[train['subarachnoid']==1]
#     train_subdural = train[train['subdural']==1]
#     list = []
#     list.append(len(train))
#     list.append(len(train_healthy))
#     list.append(len(train_epidural))
#     list.append(len(train_intraparenchymal))
#     list.append(len(train_intraventricular))
#     list.append(len(train_subarachnoid))
#     list.append(len(train_subdural))
#     return list
if __name__ == '__main__':
    if cfg.iswandb:
        wandb.init(project='ich')
        wandb_config = wandb.config
        wandb_config = wandbconfig(wandb_config, cfg)

    torch.cuda.set_device(0)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    if cfg.n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True

    path_data = cfg.dataset
    dir_train_img = os.path.join(path_data, 'train_no_brain/')
    dir_test_img = os.path.join(path_data, 'test_no_brain/')
    # train = pd.read_csv(os.path.join(path_data, 'newtrain.csv'))
    # testdf = pd.read_csv(os.path.join(path_data, 'newtest.csv'))
    train = pd.read_csv(os.path.join(path_data, 'newtrain_any.csv'))
    testdf = pd.read_csv(os.path.join(path_data, 'newtest_any.csv'))
    print('Processed img path : {}'.format(os.path.join(dir_train_img, '**.jpg')))
    png_test = glob.glob(os.path.join(dir_test_img, '*.jpg'))
    png_test = [os.path.basename(png)[:-4] for png in png_test]
    png = glob.glob(os.path.join(dir_train_img, '*.jpg'))
    png = [os.path.basename(png)[:-4] for png in png]
    print('Count of pngs : {}'.format(len(png)))
    print('Count of pngs : {}'.format(len(png_test)))
    train_imgs = set(train.Image.tolist())
    png = [p for p in png if p in train_imgs]
    print('Number of images to train on {}'.format(len(png)))
    png = np.array(png)
    png_test = np.array(png_test)
    train = train.set_index('Image').loc[png].reset_index()
    testdf = testdf.set_index('Image').loc[png_test].reset_index()
    valdf = train[train['fold'] == cfg.fold].reset_index(drop=True)
    trndf = train[train['fold'] != 4].reset_index(drop=True)  # fold 2
    print('Trn shape {} {}'.format(*trndf.shape))
    print('Tst shape {} {}'.format(*testdf.shape))
    # for debug
    # trndf = train[:1000]
    # testdf = testdf[:1000]
    # valdf = valdf[:1000]
    trnloader, valloader, tstloader = cnn_datasets.datasets(trndf, valdf, testdf, cfg)

    # 模型构建

    #       *********Mynet************
    model1 = resnet.resnet50(pretrained=True).cuda()
    model2 = torchvision.models.densenet121(pretrained=True).cuda()
    model = cnn_models.Mynet(model1, model2)

    # print(model)
    # model.classifier[6] = torch.nn.Linear(4096, 6)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()
    # model2.classifier = torch.nn.Linear(1024, 6)# nn.Sequential(*list(backbone2.children()))
    # model1.cuda()
    # model2.cuda()

    # 模型重新加载
    if cfg.load_model:  ##resume
        print(">>>>load model")
        for param in model.parameters():
            param.requires_grad = False
        input_model_file = cfg.load_path
        model.load_state_dict(torch.load(input_model_file))
        model.to(cfg.device)
    # end load

    # 参数设置
    plist = [{'params': model.parameters(), 'lr': cfg.lr}]  # 参数list
    optimizer = optim.Adam(plist, lr=cfg.lr, weight_decay=cfg.weight_dacay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_epoch, gamma=cfg.lr_gamma)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # model = nn.DataParallel(model, device_ids=[0, 1])


    print(">>>>start epoch")
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(">>>>train time: " + time)
    # tools.trainer_zhengliu(model1,model2, optims, cfg, trnloader,tstloader,sches)
    tools.trainer(model, optimizer, cfg, trnloader, tstloader, scheduler)
    print(">>>> end time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
