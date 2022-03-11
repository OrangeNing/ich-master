
import numpy as np
import sys, gc
import torch
import torch.optim as optim
import datetime
import pandas as pd
import os
from model import Mylstm, rnn_models
import ast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from datasets import lstm_datasets
from configs import config_lstm as lcfg
from utilss.utils import loadobj
from apex import amp
from trainer import tools

from utilss import evaluation
import sklearn.metrics
from collections import Counter
import warnings
from utilss import loss as los


def loademb(type, size, cnn_model):
    embpath = '/media/ps/_data1/ICH/ich-master/experiment/r50+d121+bce+1.5*(l1+l2)/emb/'
    return np.load(os.path.join(embpath, 'emb_type{}_size{}_model{}.npz'.format(type, size, cnn_model)))['arr_0']


if __name__ == '__main__':
    path_emb = '/media/ps/_data1/ICH/ich-master/experiment/r50+d121+bce+1.5*(l1+l2)/emb/'
    tstembls = [loademb('tst', 256, 'cbp')]
    tstemb = sum(tstembls) / len(tstembls)
    print('Tst shape {} {}'.format(*tstemb.shape))
    tstemb = sum(tstembls) / len(tstembls)
    tstdf = loadobj(
        os.path.join(path_emb, 'loader_type{}_size{}_model{}'.format('tst', 256, 'cbp'))).dataset.data
    tstdf['embidx'] = range(tstdf.shape[0])
    # tstdataset = lstm_datasets.IntracranialDataset(tstdf, tstemb, labels=True)
    # tstloader = DataLoader(tstdataset, batch_size=128 * 4, shuffle=False, num_workers=16, pin_memory=True)
    tstdataset = lstm_datasets.IntracranialDataset(tstdf, tstemb, labels=True)
    tstloader = DataLoader(tstdataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
    # device = torch.device('cuda')
    model = rnn_models.NeuralNet(embed_size=8192, LSTM_UNITS=2048, DO=0.2)
    modelpath = "/media/ps/_data1/ICH/ich-master/experiment/r50+d121+bce+1.5*(l1+l2)/lstm/modelBiLSTM_lossbce_epoch19.bin"
    # model.load_state_dict(torch.load(modelpath))
    model.load_state_dict(torch.load(modelpath), False)
    model.to('cuda')
    pred_list_int = []
    pred_list_sigmoid = []
    gt_list = []
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    for step, batch in enumerate(tstloader):
        labels = batch['labels'].to('cuda', dtype=torch.float)
        x = batch['emb'].to('cuda', dtype=torch.float)
        logits = model(x)
        logits = logits.to('cuda', dtype=torch.float)
        size1 = logits.size()
        labels = labels.squeeze()
        logits = logits.view(size1[0] * size1[1], -1)
        labels = labels.view(size1[0] * size1[1], -1)
        pred_out_sigmoid = torch.sigmoid(logits).cpu().detach().numpy()
        pred_out_int = np.rint(pred_out_sigmoid)
        pred_list_int.extend(pred_out_int)
        pred_list_sigmoid.extend(pred_out_sigmoid)
        gt_list.extend(labels.cpu().detach().numpy())
    f1 = evaluation.evaluation_multi(gt_list, pred_list_int, pred_out_sigmoid)
    evaluation.evaluation_everyich(gt_list, pred_list_int)
