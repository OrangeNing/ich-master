import numpy as np
import sys, gc
import torch
import torch.optim as optim
import datetime
import pandas as pd
import os

from torch.nn import init

from model import rnn_models, Mylstm
import ast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from datasets import lstm_datasets
from configs import config_lstm as lcfg
from trainer.tools import criterion_weightfocalloss
from utilss.utils import loadobj
from apex import amp
import torch.nn as nn
from trainer import tools

from utilss import evaluation
import sklearn.metrics
from collections import Counter
import warnings
from utilss import loss as los
import wandb

warnings.filterwarnings('ignore')
wandb.init(project='ich')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def wandbconfig(config):
    config.seed = lcfg.seed
    config.fold = lcfg.fold
    config.epochs = lcfg.epochs
    config.batchsize = lcfg.batchsize
    config.lr = lcfg.lr
    config.size = lcfg.size
    config.model = lcfg.lstm_model
    config.units = lcfg.lstm_units
    config.loss = lcfg.lstm_loss
    return config


def loademb(type, size, cnn_model):
    return np.load(os.path.join(path_emb, 'emb_type{}_size{}_model{}.npz'.format(type, size, cnn_model)))['arr_0']


# 1D tensor cut
def tensorCut1D(tensor, start, end):
    tensor_numpy = tensor.numpy()  # convert to numpy
    tensor_numpy = tensor_numpy[start:end]
    tensor = torch.from_numpy(tensor_numpy)
    return tensor


def tensorCut2D(tensor, list, axis):
    tensor_numpy = tensor.numpy()  # convert to numpy
    tensor_numpy = np.delete(tensor_numpy, list, axis)
    tensor = torch.from_numpy(tensor_numpy)
    return tensor


def criterion(data, targets, criterion=torch.nn.BCEWithLogitsLoss()):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    loss_all = criterion(data, targets)
    loss_any = criterion(data[:, -1:], targets[:, -1:])
    return (loss_all * 6 + loss_any * 1) / 7
    # return loss_all


def criterion_multi(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    # if lossfun == 'bce':
    loss = torch.nn.BCEWithLogitsLoss()
    loss_all = loss(data, targets)
    return loss_all
    # loss_any = loss(data[:, -1:], targets[:, -1:])
    # data = torch.sigmoid(data).data.cpu().numpy()
    # targets = targets.data.cpu().numpy()
    # loss = sklearn.metrics.hamming_loss(targets,data)


def predict(loader):
    valls = []
    imgls = []
    imgdf = loader.dataset.data.reset_index().set_index('embidx')[['Image']].copy()
    for step, batch in enumerate(loader):
        inputs = batch["emb"]
        mask = batch['mask'].to(device, dtype=torch.int)
        inputs = inputs.to(device, dtype=torch.float)
        logits = model(inputs)
        # get the mask for masked labels
        maskidx = mask.view(-1) == 1
        # reshape for
        logits = logits.view(-1, n_classes)[maskidx]
        valls.append(torch.sigmoid(logits).detach().cpu().numpy())
        # Get the list of images
        embidx = batch["embidx"].detach().cpu().numpy().astype(np.int32)
        embidx = embidx.flatten()[embidx.flatten() > -1]
        images = imgdf.loc[embidx].Image.tolist()
        imgls += images
    return np.concatenate(valls, 0), imgls


def createIndex(imgname):
    index_list = []
    for name in imgname:
        for type in label_cols:
            index_list.append(name + '_' + type)
    return index_list


def getdist(labels):
    labels = labels.squeeze()

    # y = torch.ones(size).to(device, dtype=torch.float)
    # get the mask for masked labels
    labels = labels.view(-1, n_classes)
    size = labels.size()
    dist = [0, 0, 0, 0, 0, 0]
    labels = torch.transpose(labels, 0, 1)
    i = 0
    for label in labels:
        label = label.numpy().tolist()
        count = Counter(label)[1]
        dist[i] = count / size[0]
        i = i + 1
    return dist


def criterion_weightfocalloss_lstm(dist, data, targets):
    # criterion = los.FocalLoss(alpha=0.7, gamma=2.0)
    criterion = los.WeightedFocalLoss(dist)
    loss = criterion(data, targets)
    return loss


def smoothlabel(labels, eps=0.3):
    size = labels.size()
    for i in range(size[0]):
        for j in range(size[1]):
            if labels[i][j] == 0:
                labels[i][j] = eps
            if labels[i][j] == 1:
                labels[i][j] = 1 - eps
    return labels


def relative_entropy_softmax(p1, p2):
    softmax = torch.nn.Softmax()
    p1 = torch.sigmoid(p1)
    # p2 = torch.sigmoid(p2)
    p1 = softmax(p1).cuda()
    p2 = softmax(p2).cuda()
    KL = torch.nn.KLDivLoss(reduction='batchmean').cuda()
    kl1 = KL(p1.log(), p2).cuda()
    kl2 = KL(p2.log(), p1).cuda()
    kl = (kl1 + kl2) / 2
    return kl


if __name__ == '__main__':

    wandb_config = wandb.config
    wandb_config = wandbconfig(wandb_config)

    # Print info about environments
    print('Cuda set up : time {}'.format(datetime.datetime.now().time()))
    device = torch.device('cuda')
    print('Device : {}'.format(torch.cuda.get_device_name()))
    print('Cuda available : {}'.format(torch.cuda.is_available()))
    print('Load params : time {}'.format(datetime.datetime.now().time()))
    torch.cuda.set_device(1)
    seed = int(lcfg.seed)
    size = int(lcfg.size)
    epoch = int(lcfg.epochs)
    cnn_model = lcfg.cnnmodel
    lossfun = lcfg.lstm_loss
    lr = float(lcfg.lr)
    lrgamma = float(lcfg.lrgamma)
    DECAY = float(lcfg.decay)
    batch_size = int(lcfg.batchsize)
    path_data = lcfg.dataset
    WORK_DIR = lcfg.workpath
    path_emb = lcfg.embpath
    LSTM_UNITS = int(lcfg.lstm_units)
    DROPOUT = float(lcfg.dropout)
    n_classes = 6
    label_cols = lcfg.label_cols

    # Print info about environments
    print('Cuda set up : time {}'.format(datetime.datetime.now().time()))
    # Load embeddings
    print('Load data frames...')
    # print(cnn_model)
    trndf = loadobj(
        os.path.join(path_emb, 'loader_type{}_size{}_model{}'.format('trn', size, cnn_model))).dataset.data
    tstdf = loadobj(
        os.path.join(path_emb, 'loader_type{}_size{}_model{}'.format('tst', size, cnn_model))).dataset.data
    trndf['embidx'] = range(trndf.shape[0])
    tstdf['embidx'] = range(tstdf.shape[0])
    # for debug
    # trndf  = trndf[:2000]
    # tstdf = tstdf[:2000]

    print('Load embeddings...')
    trnembls = [loademb('trn', size, cnn_model)]
    tstembls = [loademb('tst', size, cnn_model)]
    # sum_trnembles = sum(trnembls)
    # len_trnembles = len(trnembls)
    # print(trnembls)
    # print(tstembls)
    # trnemb = sum(trnembls) / len(trnembls)
    # tstemb = sum(tstembls) / len(tstembls)
    trnemb = np.array(trnembls).squeeze()
    tstemb = np.array(tstembls).squeeze()
    print(trnemb)
    print(tstemb)

    print('Trn shape {} {}'.format(*trnemb.shape))
    print('Tst shape {} {}'.format(*tstemb.shape))

    print('Create loaders...')
    trndataset = lstm_datasets.IntracranialDataset(trndf, trnemb, labels=True)
    trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=False, num_workers=16,
                           pin_memory=True)  # , collate_fn=lstm_datasets.collatefn
    tstdataset = lstm_datasets.IntracranialDataset(tstdf, tstemb, labels=True)
    tstloader = DataLoader(tstdataset, batch_size=batch_size * 4, shuffle=False, num_workers=16,
                           pin_memory=True)  # , collate_fn=lstm_datasets.collatefn

    print('Create model')
    if lcfg.lstm_model == 'BiLSTM':
        model = rnn_models.NeuralNet(embed_size=8192, LSTM_UNITS=LSTM_UNITS, DO=DROPOUT)
        # model = Mylstm.BiLSTM(LSTM_UNITS=2048, DO=0.2)
    if lcfg.lstm_model == 'GRU':
        model = rnn_models.GRU(LSTM_UNITS=LSTM_UNITS)
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    #
    # # print(param_optimizer)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    plist = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.Adam(plist, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=lrgamma)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # model = nn.DataParallel(model, device_ids=[0, 1])

    wandb.watch(model, log="all")

    for epoch in range(epoch):
        print(">>>>>start train")
        print('current epoch is {}'.format(epoch))
        tr_loss = 0.
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        for step, batch in enumerate(trnloader):
            x = batch['emb']
            y = batch['labels']
            distribution = getdist(y)
            # print(x.shape)
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)
            y = y.squeeze()
            logits = model(x)
            # y = torch.ones(size).to(device, dtype=torch.float)
            # get the mask for masked labels
            y = y.view(-1, n_classes)
            logits = logits.view(-1, n_classes)
            # loss = criterion(logits, y)
            loss = criterion_weightfocalloss(distribution, logits, y, epoch)
            # y = smoothlabel(y)
            # Get loss
            # label_smooth = smoothlabel(y)
            # logits,logits1,logits2 = model(inputs)
            # loss1 = criterion_bce(logits, label_smooth)
            # loss1 = criterion(logits,label_smooth)
            # kl1 = relative_entropy_softmax(logits, label_smooth)
            # loss1 = criterion_weightfocalloss_lstm(distribution,logits,y)
            # loss = loss1+kl1
            tr_loss += loss.item()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 500 == 0:
                print('Trn step {} of {} trn lossavg {:.5f}'. \
                      format(step, len(trnloader), (tr_loss / (1 + step))))
        scheduler.step()
        avg_loss = tr_loss / step
        print('>>>>>end train')

        print(">>>>>start test")
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        pred_list_int = []
        pred_list_sigmoid = []
        gt_list = []
        for step, batch in enumerate(tstloader):
            labels = batch['labels'].to(device, dtype=torch.float)
            x = batch['emb'].to(device, dtype=torch.float)
            logits = model(x)
            # print(logits)
            logits = logits.to(device, dtype=torch.float)
            labels = labels.squeeze()
            size1 = logits.size()
            logits = logits.view(size1[0] * size1[1], -1)
            labels = labels.view(size1[0] * size1[1], -1)
            pred_out_sigmoid = torch.sigmoid(logits).cpu().detach().numpy()
            pred_out_int = np.rint(pred_out_sigmoid)
            # print(pred_out_int)
            gt_list.extend(labels.cpu().detach().numpy())
            pred_list_int.extend(pred_out_int)
            pred_list_sigmoid.extend(pred_out_sigmoid)

        f1 = evaluation.evaluation_multi(gt_list, pred_list_int, pred_out_sigmoid, loss)
        # 评价针对于所有病灶的结果
        evaluation.evaluation_everyich(gt_list, pred_list_int)
        output_model_file = os.path.join(WORK_DIR,
                                         'model{}_loss{}_epoch{}.bin'.format(lcfg.lstm_model, lossfun, epoch, ))
        torch.save(model.state_dict(), output_model_file)
        del pred_list_int, pred_list_sigmoid, gt_list
