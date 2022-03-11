import torch
from matplotlib import pyplot as plt

from torchvision import utils as vutils
import wandb
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from pycm import *

# from scikitplot.metrics import plot_roc
wandb.init(project='ich')


def evaluation_multi(label_list, pred_int_list, pred_sigmoid_list, loss=0):
    '''
    :param label_list:
    :param pred_int_list:
    :param pred_sigmoid_list:
    :return:
    '''
    print("start testing!")
    label = np.array(label_list)
    pred_label = np.array(pred_int_list)
    # print(label.shape)
    # print(pred_label.shape)
    pred_sigmoid = np.array(pred_sigmoid_list)
    label = np.squeeze(label)
    # print(label)
    # print(pred_label)
    # auc = sklearn.metrics.roc_auc_score(pred_label,pred_sigmoid,average='micro')
    # print(auc)
    # averge_precision_slice_multi = sklearn.metrics.average_precision_score(label, pred_sigmoid)
    # f1_score_macro = sklearn.metrics.f1_score(label, pred_label, average='macro',zero_division = 0)
    # recall_score_macro = sklearn.metrics.recall_score(label, pred_label,average='macro',zero_division = 0)
    # accuracy_score_macro = sklearn.metrics.accuracy_score(label, pred_label)
    # jaccard_macro = sklearn.metrics.jaccard_score(label, pred_label,average='macro')
    # precision_score_macro = sklearn.metrics.precision_score(label, pred_label,average='macro')
    hanming_loss = round(sklearn.metrics.hamming_loss(label, pred_label), 4)
    f1_score_micro = round(sklearn.metrics.f1_score(label, pred_label, average='micro', zero_division=0), 4)
    recall_score_micro = round(sklearn.metrics.recall_score(label, pred_label, average='micro', zero_division=0), 4)
    accuracy_score_micro = round(sklearn.metrics.accuracy_score(label, pred_label), 4)
    jaccard_micro = round(sklearn.metrics.jaccard_score(label, pred_label, average='micro'), 4)
    precision_score_micro = round(sklearn.metrics.precision_score(label, pred_label, average='micro'), 4)
    print('Hanming_loss: {}'.format(hanming_loss))
    print("Accuracy_score : {}".format(accuracy_score_micro))
    print("Precision_score : {}".format(precision_score_micro))
    print("Recall_score : {}".format(recall_score_micro))
    print("F1_score:{}".format(f1_score_micro))
    print('Jaccard : {}'.format(jaccard_micro))
    print('loss : {}'.format(loss))
    print("slice end#############")
    wandb.log({
        "F1_score_slice_multi": f1_score_micro,
        "Recall_score_slice_multi": recall_score_micro,
        "Accuracy_score_slice_multi": accuracy_score_micro,
        "Precision_score_slice_multi": precision_score_micro,
        'Hamming loss': hanming_loss,
        'Jaccard': jaccard_micro,
        'loss': loss
    })
    return f1_score_micro


def evaluation_everyich(label_list_all, pred_int_list):
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    for i in range(6):
        label_list = []
        pred_list = []
        type = label_cols[i]
        for label_slice in label_list_all:
            label_list.append(label_slice[i])
        for pred_slice in pred_int_list:
            pred_list.append(pred_slice[i])

        # 评价指标   f1 recall accuracy  precision auc 对于每种病情
        f1_score = round(sklearn.metrics.f1_score(label_list, pred_list), 4)
        recall_score = round(sklearn.metrics.recall_score(label_list, pred_list), 4)
        accuracy_score = round(sklearn.metrics.accuracy_score(label_list, pred_list), 4)
        precision_score = round(sklearn.metrics.precision_score(label_list, pred_list), 4)
        cfm = confusion_matrix(label_list, pred_list)
        tn, fp, fn, tp = cfm.ravel()
        specificity = round(tn / (tn + fp), 4)
        fnr = round(fn / (tp + fn), 4)
        fpr = round(fp / (fp + tn), 4)
        print("F1_score_{}:{}".format(type, f1_score))
        print("Accuracy_score_{}:{}".format(type, accuracy_score))
        print("Precision_score_{}:{}".format(type, precision_score))
        print("Recall_score_{}:{}".format(type, recall_score))  # = sensitivity = TPR
        print('Specificity_{}:{}'.format(type, specificity))  # TNR
        print("FPR: {}".format(fpr))  # FPR
        print("FNR: {}".format(fnr))  # FNR
        print("cfm: {}".format(cfm))
        print("#############")

        # wandb.log({
        #     "F1_score_type_{}".format(type): f1_score,
        #     "Recall_score_type_{}".format(type): recall_score,
        #     "Accuracy_score_type_{}".format(type): accuracy_score,
        #     "Specificity_{}".format(type): Specificity,
        #     "Averge_precision_type_{}".format(type): precision_score,
        #     "Auc_{}".format(type):auc
        # })


def evaluation_multi_nowandb(label_list, pred_int_list, pred_sigmoid_list, loss=0):
    '''
    :param label_list:
    :param pred_int_list:
    :param pred_sigmoid_list:
    :return:
    '''
    print("start testing!")
    label = np.array(label_list)
    pred_label = np.array(pred_int_list)
    pred_sigmoid = np.array(pred_sigmoid_list)
    label = np.squeeze(label)
    # print(label.shape)
    # print(pred_label.shape)
    # print(label)
    # print(pred_label)
    hanming_loss = round(sklearn.metrics.hamming_loss(label, pred_label), 4)
    # auc = sklearn.metrics.roc_auc_score(pred_label,pred_sigmoid,average='micro')
    # print(auc)
    # averge_precision_slice_multi = sklearn.metrics.average_precision_score(label, pred_sigmoid)
    # f1_score_macro = sklearn.metrics.f1_score(label, pred_label, average='macro',zero_division = 0)
    # recall_score_macro = sklearn.metrics.recall_score(label, pred_label,average='macro',zero_division = 0)
    # accuracy_score_macro = sklearn.metrics.accuracy_score(label, pred_label)
    # jaccard_macro = sklearn.metrics.jaccard_score(label, pred_label,average='macro')
    # precision_score_macro = sklearn.metrics.precision_score(label, pred_label,average='macro')
    f1_score_micro = round(sklearn.metrics.f1_score(label, pred_label, average='micro', zero_division=0), 4)
    recall_score_micro = round(sklearn.metrics.recall_score(label, pred_label, average='micro', zero_division=0), 4)
    accuracy_score_micro = round(sklearn.metrics.accuracy_score(label, pred_label), 4)
    jaccard_micro = round(sklearn.metrics.jaccard_score(label, pred_label, average='micro'), 4)
    precision_score_micro = round(sklearn.metrics.precision_score(label, pred_label, average='micro'), 4)
    # specificity_score_micro = round(spe(label, pred_label, 6), 4)
    # cfm = sklearn.metrics.confusion_matrix(label.ravel(), pred_label.ravel())
    # print("cfm: {}".format(cfm))
    print('Hanming_loss: {}'.format(hanming_loss))
    print("F1_score:{}".format(f1_score_micro))
    print("Recall_score : {}".format(recall_score_micro))
    print("Accuracy_score : {}".format(accuracy_score_micro))
    print("Precision_score : {}".format(precision_score_micro))
    print('Jaccard : {}'.format(jaccard_micro))
    # print('Specificity:{}'.format(specificity))
    # print("tn:{},fp:{},fn:{},tp:{}".format(tn, fp, fn, tp))
    print('loss : {}'.format(loss))
    print("slice end#############")
    return f1_score_micro


def evaluation_everyich_nowandb(label_list_all, pred_int_list):
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    for i in range(6):
        label_list = []
        pred_list = []
        type = label_cols[i]
        for label_slice in label_list_all:
            label_list.append(label_slice[i])
        for pred_slice in pred_int_list:
            pred_list.append(pred_slice[i])
        # 评价指标   f1 recall accuracy  precision auc 对于每种病情
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_list, pred_list)
        auc = round(sklearn.metrics.auc(fpr, tpr), 4)
        f1_score = round(sklearn.metrics.f1_score(label_list, pred_list), 4)
        recall_score = round(sklearn.metrics.recall_score(label_list, pred_list), 4)
        accuracy_score = round(sklearn.metrics.accuracy_score(label_list, pred_list), 4)
        precision_score = round(sklearn.metrics.precision_score(label_list, pred_list), 4)
        cfm = confusion_matrix(label_list, pred_list)
        tn, fp, fn, tp = cfm.ravel()
        specificity = round(tn / (tn + fp), 4)
        fnr = round(fn / (tp + fn), 4)
        fpr = round(fp / (fp + tn), 4)
        print("F1_score_{}:{}".format(type, f1_score))
        print("Recall_score_{}:{}".format(type, recall_score))
        print('Specificity_{}:{}'.format(type, specificity))
        print("Accuracy_score_{}:{}".format(type, accuracy_score))
        print("Precision_score_{}:{}".format(type, precision_score))
        print("FPR: {}".format(fpr))  # FPR
        print("FNR: {}".format(fnr))  # FNR
        print("cfm: {}".format(cfm))
        print("#############")


def evaluation_everyich_auc(label_list_all, pred_int_list):
    label_cols = ['healthy', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    for i in range(4):
        i = i + 1
        label_list = []
        pred_list = []
        type = label_cols[i]
        for label_slice in label_list_all:
            label_list.append(label_slice[0][i])
        for pred_slice in pred_int_list:
            pred_list.append(pred_slice[i])
        label_list = np.array(label_list)
        pred_list = np.array(pred_list)
        # 评价指标   f1 recall accuracy  precision auc 对于每种病情
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_list, pred_list)
        # print(fpr)
        # print(tpr)
        # sensitivity = round(tpr,4)
        # specificity = round(fpr,4)
        auc = round(sklearn.metrics.auc(fpr, tpr), 4)
        # sensitivity = round(sklearn.metrics.accuracy_score)
        # f1_score = round(sklearn.metrics.f1_score(label_list, pred_list),4)
        recall_score = round(sklearn.metrics.recall_score(label_list, pred_list), 4)
        accuracy_score = round(sklearn.metrics.accuracy_score(label_list, pred_list), 4)
        specificity = round(sklearn.metrics.accuracy_score(label_list, pred_list), 4)
        # precision_score = round(sklearn.metrics.f1_score(label_list, pred_list),4)
        print("sensitivity{}:{}".format(type, recall_score))
        print("specificity{}:{}".format(type, specificity))
        print("Accuracy_score_{}:{}".format(type, accuracy_score))
        print('Auc_{}:{}'.format(type, auc))
        print("#############")
        # wandb.log({
        #     "F1_score_type_{}".format(type): sensitivity,
        #     "Recall_score_type_{}".format(type): specificity,
        #     "Accuracy_score_type_{}".format(type): accuracy_score,
        #     "Auc_{}".format(type):auc
        # })
