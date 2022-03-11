import os
from configs import config_cnn as cfg
import torch
from torch.backends import cudnn
import torch.nn as nn
import glob
import numpy as np
import pandas as pd
import torch.optim as optim
from apex import amp
import datetime
from model import cnn_models, pvt
import warnings
from datasets import cnn_datasets
import torch.nn
# import wandb
# wandb.init(project='ich')
import torchvision
from model import resnet
from utilss import utils
from efficientnet_pytorch import EfficientNet

warnings.filterwarnings('ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# def wandbconfig(config):
#     config.seed = cfg.seed
#     config.fold = cfg.fold
#     config.epochs = cfg.epochs
#     config.batchsize = cfg.batchsize
#     config.lr = cfg.lr
#     config.size = cfg.size
#     config.hflip = cfg.hflip  # Augmentation - Embedding horizontal flip
#     config.transpose = cfg.transpose
#     config.autocrop = cfg.autocrop
#     config.loss = cfg.loss
#     config.model = cfg.model
#     return config


if __name__ == '__main__':
    print("start time :{}".format(datetime.datetime.now()))
    # print(torch.cuda.device_count())
    # wandb_config = wandb.config
    # wandb_config = wandbconfig(wandb_config)
    np.random.seed(cfg.seed)
    # torch.cuda.set_device(1)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    if cfg.n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # 数据加载
    path_data = cfg.dataset
    dir_train_img = os.path.join(path_data, 'train_no_brain/')
    dir_test_img = os.path.join(path_data, 'test_no_brain/')
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
    # trndf = train[:200]
    # testdf = testdf[:200]
    # valdf = valdf[:200]
    trnloader, valloader, tstloader = cnn_datasets.datasets_emb(trndf, valdf, testdf, cfg)

    # 加载模型
    print(">>>>load model")

    model1 = resnet.resnet50(pretrained=True)
    model2 = torchvision.models.densenet121(pretrained=True)
    model = cnn_models.Mynet(model1, model2)

    # model = EfficientNet.from_pretrained('efficientnet-b6')
    # model._fc = nn.Linear(model._fc.in_features, cfg.n_classes)

    # model = pvt.pvt_medium(pretrained=True)

    # model = nn.DataParallel(model,device_ids=[0,1])

    for param in model.parameters():
        param.requires_grad = False
    input_model_file = cfg.load_path  # 权重文件地址
    model_file = torch.load(input_model_file)
    model.load_state_dict(model_file)
    model.cuda()

    plist = [{'params': model.parameters(), 'lr': cfg.lr}]  # 参数list
    optimizer = optim.Adam(plist, lr=cfg.lr, weight_decay=cfg.weight_dacay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_epoch, gamma=cfg.lr_gamma)  # 在迭代过程中按step调整lr（下降）
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 大写“欧1”

    # model.load_state_dict({k.replace('module.', ''): v for k, v in model_file.items()})
    # plist = [{'params': model.parameters(), 'lr': cfg.lr}]

    print(">>>>end model")
    # end load

    # wandb.watch(model,log="all")
    # torch.backends.cudnn.enabled= False

    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    model.eval()
    DATASETS = ['tst', 'trn']
    LOADERS = [tstloader, trnloader]

    print("start emb")
    for typ, loader in zip(DATASETS, LOADERS):
        ls = []
        for step, batch in enumerate(loader):
            if step % 1000 == 0:
                print('Embedding {} step {} of {}'.format(typ, step, len(loader)))
            inputs = batch["image"]
            inputs = inputs.to('cuda', dtype=torch.float)
            # print(inputs)
            out = model.features(inputs)  # 只取cbp之后的emb
            # out = model(inputs)
            # print(out)
            ls.append(out.detach().cpu().numpy())
        outemb = np.concatenate(ls, 0).astype(np.float32)
        print('Write embeddings : type {} shape {} {}'.format(typ, *outemb.shape))
        print(outemb)
        fembname = 'emb_type{}_size{}_model{}'.format(typ, cfg.size, cfg.model)
        print('Embedding file name : {}'.format(fembname))
        np.savez_compressed(os.path.join(cfg.emb_pth, fembname), outemb)
        print("Write loders: type {}".format(typ))
        utils.dumpobj(
            os.path.join(cfg.emb_pth, 'loader_type{}_size{}_model{}'.format(typ, cfg.size, cfg.model)), loader)
        print("end time :{}".format(datetime.datetime.now()))