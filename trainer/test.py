
import os

import sys
import torchvision
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest101

from configs import config_cnn as cfg
# import configs.config_cnn as cfg
from trainer import tools
import torch
import torch.nn as nn
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
from model import cnn_models, rnn_models, pvt
from utilss import loss
from utilss import evaluation
import warnings
from datasets import cnn_datasets
import sys
import wandb
from model import resnet

wandb.init(project='ich')


# warnings.filterwarnings('ignore')
def test(model, tstloder, loss=0):
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    # print(">>>>INFER IS PRED")
    pred_sigmoid_list = []
    pred_int_list = []
    label_list = []
    pred_list_binary = []
    label_list_binary = []
    # model = torch.nn.DataParallel(model, device_ids=list(range(cfg.n_gpu)))
    for step, batch in enumerate(tstloder):
        # try:
        inputs = batch["image"]
        inputs = inputs.to(cfg.device, dtype=torch.float)
        labels = batch["labels"]
        labels = labels.to(cfg.device, dtype=torch.float)
        labels = labels.squeeze()
        logits = model(inputs)
        logits = logits[0]  # 在test其他模型的时候，需要注释掉
        pred_out_sigmoid = torch.sigmoid(logits).cpu().numpy()
        pred_out_int = np.rint(pred_out_sigmoid)
        labels = np.array(labels.cpu())
        ###转化成单标签分类
        pred_out_binary = np.rint(pred_out_int.ravel().tolist())
        labels_binary = labels.ravel().tolist()
        pred_list_binary.extend(pred_out_binary)
        label_list_binary.extend(labels_binary)
        ##多标签分类
        label_list.extend(labels)
        pred_int_list.extend(pred_out_int)
        pred_sigmoid_list.extend(pred_out_sigmoid)
        # except:
        #     print(inputs.size())
        #     print(batch['labels'].size())
    # 评价多标签分类的结果slice
    f1 = evaluation.evaluation_multi(label_list, pred_int_list, pred_out_sigmoid, loss)
    # 评价针对于所有病灶的结果
    evaluation.evaluation_everyich(label_list, pred_int_list)
    return f1


def test_nowandb(model, tstloder, loss=0):
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    # print(">>>>INFER IS PRED")
    pred_sigmoid_list = []
    pred_int_list = []
    label_list = []
    pred_list_binary = []
    label_list_binary = []
    # model = torch.nn.DataParallel(model, device_ids=list(range(cfg.n_gpu)))
    for step, batch in enumerate(tstloder):
        # try:
        inputs = batch["image"]
        inputs = inputs.to(cfg.device, dtype=torch.float)
        labels = batch["labels"]
        labels = labels.to(cfg.device, dtype=torch.float)
        labels = labels.squeeze()
        logits = model(inputs)
        pred_out_sigmoid = torch.sigmoid(logits).cpu().numpy()
        pred_out_int = np.rint(pred_out_sigmoid)
        labels = np.array(labels.cpu())
        ###转化成单标签分类
        pred_out_binary = np.rint(pred_out_int.ravel().tolist())
        labels_binary = labels.ravel().tolist()
        pred_list_binary.extend(pred_out_binary)
        label_list_binary.extend(labels_binary)
        ##多标签分类
        label_list.extend(labels)
        pred_int_list.extend(pred_out_int)
        pred_sigmoid_list.extend(pred_out_sigmoid)
        # except:
        #     print(inputs.size())
        #     print(batch['labels'].size())
    # 评价多标签分类的结果slice
    f1 = evaluation.evaluation_multi_nowandb(label_list, pred_int_list, pred_out_sigmoid, loss)
    # 评价针对于所有病灶的结果
    # evaluation.evaluation_everyich_nowandb(label_list, pred_int_list)
    # return f1
    # evaluation(label_list,pred_int_list,pred_sigmoid_list,label_list_binary,pred_list_binary)


def test_zhengliu(model1, model2, tstloder, loss=0):
    model1 = model1.eval()
    model2 = model2.eval()
    # print(">>>>INFER IS PRED")
    pred_sigmoid_list = []
    pred_int_list = []
    label_list = []
    pred_list_binary = []
    label_list_binary = []
    for param in model1.parameters():
        param.requires_grad = False
    for param in model2.parameters():
        param.requires_grad = False
    # model = torch.nn.DataParallel(model, device_ids=list(range(cfg.n_gpu)))
    for step, batch in enumerate(tstloder):
        # try:
        inputs = batch["image"]
        inputs = inputs.to(cfg.device, dtype=torch.float)
        labels = batch["labels"]
        labels = labels.to(cfg.device, dtype=torch.float)
        # logits = model(inputs)
        # logits, output2, output3 = model(inputs)
        logits1 = model1(inputs)
        logits2 = model2(inputs)
        pred_out1 = torch.sigmoid(logits1)
        pred_out2 = torch.sigmoid(logits2)
        pred_out_sigmoid = ((pred_out1 + pred_out2) / 2).cpu().detach().numpy()
        # pred_out_sigmoid = torch.sigmoid(logits).cpu().numpy()
        pred_out_int = np.rint(pred_out_sigmoid)
        labels = np.array(labels.cpu())
        ###转化成单标签分类
        pred_out_binary = np.rint(pred_out_int.ravel().tolist())
        labels_binary = labels.ravel().tolist()
        pred_list_binary.extend(pred_out_binary)
        label_list_binary.extend(labels_binary)
        ##多标签分类
        label_list.extend(labels)
        pred_int_list.extend(pred_out_int)
        pred_sigmoid_list.extend(pred_out_sigmoid)
        # except:
        #     print(inputs.size())
        #     print(batch['labels'].size())
    # 评价多标签分类的结果slice
    f1 = evaluation.evaluation_multi(label_list, pred_int_list, pred_out_sigmoid, loss)
    # 评价针对于所有病灶的结果
    evaluation.evaluation_everyich(label_list, pred_int_list)
    return f1


if __name__ == '__main__':

    os.environ['CUDA_VISION_DEVICES'] = '1'
    model_type = 'LSTM'
    if model_type == 'CNN':
        '''cnn'''
        model = EfficientNet.from_pretrained('efficientnet-b6')
        model._fc = nn.Linear(model._fc.in_features, cfg.n_classes)
        modelpath = "/media/ps/_data1/ICH/ich-master/experiment/10-3-efficientnet-b6/weights/best_model.bin"
        # model.load_state_dict(torch.load(modelpath))
        model_file = torch.load(modelpath)
        model.load_state_dict({k.replace('module.', ''): v for k, v in model_file.items()})
    if model_type == 'LSTM':
        '''lstm'''
        model = rnn_models.NeuralNet(embed_size=8192, LSTM_UNITS=2048, DO=0.2)
        modelpath = "/media/ps/_data1/ICH/ich-master/experiment/r50+d121+bce+1.5*(l1+l2)/lstm/modelBiLSTM_lossbce_epoch20.bin"
        model_file = torch.load(modelpath)
        model.load_state_dict(model_file)
    model.to(cfg.device)
    path_data = '/media/ps/_data1/ICH/ich-master/dataset/'
    testdf = pd.read_csv(os.path.join(path_data, 'newtest_any.csv'))
    dir_test_img = os.path.join(path_data, 'test_no_brain/')
    png_test = glob.glob(os.path.join(dir_test_img, '*.jpg'))
    png_test = [os.path.basename(png)[:-4] for png in png_test]
    print('Count of test pngs : {}'.format(len(png_test)))
    png_test = np.array(png_test)
    testdf = testdf.set_index('Image').loc[png_test].reset_index()
    print('Tst shape {} {}'.format(*testdf.shape))
    ##for debug
    tstloader = cnn_datasets.datasets_test(testdf, dir_test_img, cfg)
    test_nowandb(model, tstloader, 0)
