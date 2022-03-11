
import os
import wandb

from configs import config_cnn as cfg
# import configs.config_cnn as cfg
from trainer import tools
import torch
from torch.backends import cudnn

import glob
import numpy as np
import pandas as pd
import torch.optim as optim
from apex import amp
import ast
from utilss import test_csv_any
from torch.utils.data import DataLoader
import datetime
from model import cnn_models
from utilss import loss

import warnings
from datasets import  cnn_datasets
import sys
# import wandb
# wandb.init(project='ich')
# warnings.filterwarnings('ignore')
def wandbconfig(config):
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
    return config
if __name__ == '__main__':
    # wandb_config = wandb.config
    # wandb_config = wandbconfig(wandb_config)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    if cfg.n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    path_data = cfg.dataset
    dir_train_img = os.path.join(path_data, 'dataset/train/')
    dir_test_img = os.path.join(path_data, 'dataset/test/')
    train = pd.read_csv(os.path.join(path_data, 'mytrain_new.csv'))
    testdf = pd.read_csv(os.path.join(path_data, 'mytest_new.csv'))
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
    valdf = train[train['fold']== cfg.fold].reset_index(drop=True)
    trndf = train[train['fold']== cfg.fold + 1].reset_index(drop=True)#fold 2
    print('Trn shape {} {}'.format(*trndf.shape))
    print('Tst shape {} {}'.format(*testdf.shape))
    # for debug
    # trndf = train[:200]
    # testdf = testdf[:200]
    # valdf = valdf[:200]

    ##seq
    # trnmdf = pd.read_csv(os.path.join(path_data, 'new_train_metadata_csv_fold2.csv'))
    # tstmdf = pd.read_csv(os.path.join(path_data, 'new_test_metadata.csv'))
    # trnmdf['SliceID'] = trnmdf[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(
    #     lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
    # tstmdf['SliceID'] = tstmdf[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(
    #     lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
    #
    # poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
    # trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient'] \
    #                                .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
    # tstmdf[poscols] = pd.DataFrame(tstmdf['ImagePositionPatient'] \
    #                                .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
    #
    # trnmdf = trnmdf.sort_values(['SliceID'] + poscols) \
    #     [['PatientID', 'SliceID', 'SOPInstanceUID'] + poscols].reset_index(drop=True)
    # tstmdf = tstmdf.sort_values(['SliceID'] + poscols) \
    #     [['PatientID', 'SliceID', 'SOPInstanceUID'] + poscols].reset_index(drop=True)
    #
    # trnmdf['seq'] = (trnmdf.groupby(['SliceID']).cumcount() + 1)
    # tstmdf['seq'] = (tstmdf.groupby(['SliceID']).cumcount() + 1)
    #
    # keepcols = ['PatientID', 'SliceID', 'SOPInstanceUID', 'seq']
    # trnmdf = trnmdf[keepcols]  #  PattientID SliceId instanceID seq
    # tstmdf = tstmdf[keepcols]  #  PattientID SliceId instanceID seq
    # trnmdf.columns = tstmdf.columns = ['PatientID', 'SliceID', 'Image', 'seq']
    # trnmdf = trnmdf.drop('PatientID',axis=1)
    # trnmdf = trnmdf.drop('SliceID',axis=1)
    # trndf_addseq = pd.merge(trndf,trnmdf,on= 'Image')
    # tstmdf = tstmdf.drop('PatientID', axis=1)
    # tstmdf = tstmdf.drop('SliceID', axis=1)
    # tstmdf_addseq = pd.merge(testdf, tstmdf, on='Image')
    ###end seq
    tstloader = cnn_datasets.datasets(testdf,cfg)
    model = cnn_models.model_pick(cfg.model)
    if cfg.load_model:##resume
        print(">>>>load model")
        for param in model.parameters():
            param.requires_grad = False
        input_model_file = cfg.load_path
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg.n_gpu)))
        model.load_state_dict(torch.load(input_model_file))
        model.to(cfg.device)
    # end load
    plist = [{'params': model.parameters(), 'lr': cfg.lr}]
    optimizer = optim.Adam(plist, lr=cfg.lr)
    model.to(cfg.device)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 01
    model = torch.nn.DataParallel(model, device_ids=list(range(cfg.n_gpu)))
    wandb.watch(model,log="all")
    print(">>>>start epoch")
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if cfg.infer == 'PRED':
        model.to(cfg.device)
        tester = tools.preder(model,tstloader,cfg)







