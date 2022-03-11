import torch
from utilss import test_csv_any
from utilss import evaluation_mullabel as eval
import os
import numpy as np
from apex import amp
import pandas as pd
import wandb
import sklearn.metrics
from trainer import test
from utilss import loss as los
from collections import Counter
from utilss.Sinkhorn import SinkhornDistance
from geomloss import SamplesLoss
import tensorflow as tf


# wandb.init(project='ich')
# def criterion(data,targets,criterion = loss.WieghtBinaryEntropyLoss()):#torch.nn.BCEWithLogitsLoss()
#     loss = criterion(data, targets)
#     return loss

def criterion_bce(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    # if lossfun == 'bce':
    ##原来的
    # torch.optim.Adam()
    loss = torch.nn.BCEWithLogitsLoss()
    loss_all = loss(data, targets)
    loss_any = loss(data[:, -1:], targets[:, -1:])
    return (loss_all * 6 + loss_any * 1) / 7
    # 现在的softmax
    # loss = los.multilabel_categorical_crossentropy(data,targets)
    return loss


def criterion_weightfocalloss(dist, data, targets, epoch):
    # criterion = los.FocalLoss(alpha=0.7, gamma=2.0)
    criterion = los.WeightedFocalLoss(dist, epoch)
    loss = criterion(data, targets)
    return loss


def criterion_weightfocalloss_v2(dist, data, targets, epoch):
    # criterion = los.FocalLoss(alpha=0.7, gamma=2.0)
    criterion = los.WeightedFocalLoss_v2(dist)
    loss = criterion(data, targets, epoch)
    return loss


def getdist(labels):
    labels = labels.squeeze()
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
    kl = KL(p2.log(), p1).cuda()  # p2是真实分布,p1是理论分布，让p1更接近于p2
    return kl


def relative_entropy_softmaxlabel(p1, p2):
    softmax = torch.nn.Softmax()
    p1 = torch.sigmoid(p1)
    p1 = softmax(p1).cuda()
    p2 = softmax(p2).cuda()
    KL = torch.nn.KLDivLoss(reduction='batchmean').cuda()
    kl = KL(p1.log(), p2).cuda()
    return kl


def trainer_zhengliu(model1, model2, optimizer, cfg, trnloader, tstloader, scheduler):
    print(">>>>INFER IS TRAIN ")
    for epoch in range(cfg.epochs):
        if cfg.load_model:
            epoch = epoch + cfg.load_start
        print('Epoch {}/{}'.format(epoch, cfg.epochs - 1))
        print('-' * 10)
        print(">>>start train")
        ''''''
        # if cfg.freeze:
        #     for param in model.fc.parameters():
        #         param.requires_grad = True
        # else:
        for param in model1.parameters():
            param.requires_grad = True
        for param in model2.parameters():
            param.requires_grad = True
        model1.train()
        model2.train()
        tr_loss = 0
        for step, batch in enumerate(trnloader):
            if step % 1000 == 0:
                print('Train step {} of {}'.format(step, len(trnloader)))
            # try:
            inputs = batch["image"]
            labels = batch["labels"]
            distribution = getdist(labels)
            inputs = inputs.to(cfg.device, dtype=torch.float)
            labels = labels.to(cfg.device, dtype=torch.float)
            labels = labels.squeeze()
            logits1 = model1(inputs)
            logits2 = model2(inputs)
            output1 = torch.softmax(logits1).cuda()
            output2 = torch.softmax(logits2).cuda()
            kl1 = relative_entropy_softmax(output1, output2)
            loss1 = criterion_weightfocalloss(distribution, logits1, labels, epoch) + kl1
            loss1 = loss1.requires_grad_()
            loss1.backward()
            optimizer[0].step()
            optimizer[0].zero_grad()
            scheduler[0].step()
            logits1 = model1(inputs)
            logits2 = model2(inputs)
            output1 = torch.sigmoid(logits1).cuda()
            output2 = torch.sigmoid(logits2).cuda()
            kl2 = relative_entropy(output2, output1)
            loss2 = criterion_weightfocalloss(distribution, logits2, labels, epoch) + kl2
            print('kl1', kl1)
            print('loss1', loss1)
            print('kl2', kl2)
            print('loss2', loss2)
            loss2 = loss2.requires_grad_()
            loss2.backward()
            optimizer[1].step()
            optimizer[1].zero_grad()
            scheduler[1].step()
            loss = loss1 + loss2
            if step % 1000 == 0:
                print('The loss of train step {} of {} is {}'.format(step, len(trnloader), loss.item()))
            tr_loss += loss.item()
        epoch_loss_avg = tr_loss / len(trnloader)
        print('Training Loss: {:.4f}'.format(epoch_loss_avg))
        output_model_file_model1 = os.path.join(cfg.model_save_path,
                                                'weights/model{}_{}_epoch{}.bin'.format(cfg.model1, cfg.size, epoch))
        output_model_file_model2 = os.path.join(cfg.model_save_path,
                                                'weights/model{}_{}_epoch{}.bin'.format(cfg.model2, cfg.size, epoch))
        torch.save(model1.state_dict(), output_model_file_model1)
        torch.save(model2.state_dict(), output_model_file_model2)
        f1 = test.test_zhengliu(model1, model2, tstloader, epoch_loss_avg)


def trainer(model, optimizer, cfg, trnloader, tstloader, scheduler):
    print(">>>>INFER IS TRAIN ")
    epoch_loss_min = 10000
    best_epoch = 0
    best_model = model
    for epoch in range(cfg.epochs):
        if cfg.load_model:
            epoch = epoch + cfg.load_start
        print('Epoch {}/{}'.format(epoch, cfg.epochs - 1))
        print('-' * 10)
        print(">>>start train")
        ''''''
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        tr_loss = 0
        for step, batch in enumerate(trnloader):
            if step % 1000 == 0:
                print('Train step {} of {}'.format(step, len(trnloader)))
            # try:
            inputs = batch["image"]
            labels = batch["labels"]
            distribution = getdist(labels)
            inputs = inputs.to(cfg.device, dtype=torch.float)
            labels = labels.to(cfg.device, dtype=torch.float)
            labels = labels.squeeze()
            logits_all = model(inputs)
            label_smooth = smoothlabel(labels)
            # resnet50+mwf
            if cfg.train_strategy == 'single':
                logits = logits_all
                loss_mwf = criterion_weightfocalloss(distribution, logits, label_smooth, epoch).cuda()
                loss_srn = relative_entropy_softmax(logits, label_smooth).cuda()
                loss = loss_mwf + 2 * loss_srn
            if cfg.train_strategy == 'all':
                logits = logits_all[0]
                logit_net1 = logits_all[1]
                logit_net2 = logits_all[2]
                loss_mwf_net1 = criterion_weightfocalloss(distribution, logit_net1, label_smooth,
                                                          epoch).cuda()  # net1是resnet50
                loss_mwf_net2 = criterion_weightfocalloss(distribution, logit_net2, label_smooth,
                                                          epoch).cuda()  # net2是densenet121
                # loss_srn = relative_entropy_softmax(logits, label_smooth)
                loss_srn_net1 = relative_entropy_softmax(logit_net1, label_smooth).cuda()
                loss_srn_net2 = relative_entropy_softmax(logit_net2, label_smooth).cuda()
                loss_bce = criterion_bce(logits, labels).cuda()
                loss = loss_bce + 1.5 * (loss_mwf_net1 + loss_mwf_net2 + loss_srn_net1 + 0.5 * loss_srn_net2)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 1000 == 0:
                print('The loss of train step {} of {} is {}'.format(step, len(trnloader), loss.item()))
            tr_loss += loss.item()
        epoch_loss_avg = tr_loss / len(trnloader)

        # 保存所有epoch中最好的一次的参数
        if epoch_loss_min > epoch_loss_avg:
            epoch_loss_min = epoch_loss_avg
            best_model = model
            best_epoch = epoch
        print('Training Loss: {:.4f}'.format(epoch_loss_avg))
        output_model_file = os.path.join(cfg.model_save_path,
                                         'weights/model{}_{}_epoch{}.bin'.format(cfg.model, cfg.size, epoch))
        torch.save(model.state_dict(), output_model_file)  # 保存模型的状态字典
        scheduler.step()  # 表示一次epoch迭代，这里就会更新学习率
        f1 = test.test(model, tstloader, epoch_loss_avg)
    output_model_file = os.path.join(cfg.model_save_path, 'weights/best_model.bin')
    torch.save(best_model.state_dict(), output_model_file)
    # wandb.save("best_model")
    print("the best epoch is {}".format(best_epoch))


def evaluation(label_list, pred_int_list, pred_sigmoid_list, label_list_binary, pred_list_binary, epoch_loss_avg):
    print("start testing!")
    ###slice
    print("slice eval############")
    ###binary
    f1_score_slice_binary, recall_score_slice_binary, accuracy_score_slice_binary, precision_score_slice_binary, auc_slice_binary = eval.binary_metrics(
        label_list_binary, pred_list_binary)
    print("F1_score_slice_binary is {}".format(f1_score_slice_binary))
    print("Recall_score_slice_binary is {}".format(recall_score_slice_binary))
    print('Accuracy_score_slice_binary is {}'.format(accuracy_score_slice_binary))
    print('Averge_precision_slice_binary is {}'.format(precision_score_slice_binary))
    print('AUC_slice_binary is {}'.format(auc_slice_binary))
    print("---------------------------------------")
    ###slice multi###
    label = np.array(label_list)
    pred_label = np.array(pred_int_list)
    pred_sigmoid = np.array(pred_sigmoid_list)
    label = np.squeeze(label)

    precision_slice_multi = sklearn.metrics.precision_score(label, pred_label, average='macro')
    f1_score_multi_samples_slice_multi = sklearn.metrics.f1_score(label, pred_label, average='macro', zero_division=0)
    recall_score_slice_multi = sklearn.metrics.recall_score(label, pred_label, average='macro', zero_division=0)
    accuracy_score_slice_multi = sklearn.metrics.accuracy_score(label, pred_label)
    jaccard = sklearn.metrics.jaccard_score(label, pred_label, average='macro')
    hanming_loss = sklearn.metrics.hamming_loss(label, pred_label)
    confusion_matrix = sklearn.metrics.confusion_matrix(label, pred_label)


    print("F1_score_slice_multi is {}".format(f1_score_multi_samples_slice_multi))
    print("Recall_score_slice_multi is {}".format(recall_score_slice_multi))
    print("Accuracy_score_slice_multi is {}".format(accuracy_score_slice_multi))
    print("Precision_slice_multi is {}".format(precision_slice_multi))
    print("Hanming_loss_multi is {}".format(hanming_loss))
    print("Jaccard_multi is {}".format(jaccard))
    print("slice end#############")
    wandb.log({
        "F1_score_slice_binary": f1_score_slice_binary,
        "Recall_score_slice_binary": recall_score_slice_binary,
        "Accuracy_score_slice_binary": accuracy_score_slice_binary,
        "Averge_precision_slice_binary": precision_score_slice_binary,
        "AUC_slice_binary": auc_slice_binary,
        "F1_score_slice_multi": f1_score_multi_samples_slice_multi,
        "Recall_score_slice_multi": recall_score_slice_multi,
        "Accuracy_score_slice_multi": accuracy_score_slice_multi,
        "Averge_precision_slice_multi": precision_slice_multi,
        'Hanming_loss_multi': hanming_loss,
        'Jaccard_multi': jaccard,
        "Epoch_loss": epoch_loss_avg,
    })


def lstm_evaluation(label_case, pred_sigmoid_case, pred_label_case, epoch_loss):
    # case_label_list, case_pred_list, case_logit_list =  test_csv_any.test_case(label_case, pred_label_case,pred_sigmoid_case)
    label_case = np.array(label_case)
    pred_label_case = np.array(pred_label_case)
    pred_sigmoid_case = np.array(pred_sigmoid_case)
    hamming_loss = sklearn.metrics.hamming_loss(label_case, pred_label_case)
    rank_loss = sklearn.metrics.label_ranking_loss(label_case, pred_label_case)
    f1_score_samples = sklearn.metrics.f1_score(label_case, pred_label_case, average='samples', zero_division=0)
    averge_precision_case = sklearn.metrics.average_precision_score(label_case, pred_label_case)
    accuracy_score_case_multi = sklearn.metrics.accuracy_score(label_case, pred_label_case)
    recall_score_case_multi = sklearn.metrics.recall_score(label_case, pred_label_case, average='samples',
                                                           zero_division=0)
    print("Hamming_loss_case is {}".format(hamming_loss))
    print("F1_score_case is {}".format(f1_score_samples))
    print("Recall_score_case is {}".format(recall_score_case_multi))
    print('Accuracy_score_case is {}'.format(accuracy_score_case_multi))
    print('Averge_precision_case  is {}'.format(averge_precision_case))
    print("Rank_loss_case is {}".format(rank_loss))
    # print('Auc_case is {}'.format(auc))
    # wandb.log({
    #     "Hamming_loss_case":hamming_loss,
    #     "Rank_loss_case ":rank_loss,
    #     "F1_score_case": f1_score_samples,
    #     "Recall_score_case": recall_score_case_multi,
    #     "Accuracy_score_case": accuracy_score_case_multi,
    #     "Averge_precision_case": averge_precision_case,
    #     "Epoch_loss":epoch_loss,
    # })


def createIndex(imgname, cfg):
    index_list = []
    for name in imgname:
        for type in cfg.label_cols:
            index_list.append(name + '_' + type)
    return index_list


def preder_test(model, tstloader, cfg, epoch_loss_avg):
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
    for step, batch in enumerate(tstloader):
        # try:
        inputs = batch["image"]
        inputs = inputs.to(cfg.device, dtype=torch.float)
        labels = batch["labels"]
        labels = labels.to(cfg.device, dtype=torch.float)
        # inputs = inputs.squeeze()
        # labels = labels.squeeze()
        # inputs_size = inputs.size()
        # labels_size = labels.size()
        # inputs = torch.reshape(inputs,
        #                        (inputs_size[0] * inputs_size[1], inputs_size[2], inputs_size[3], inputs_size[4]))
        # labels = torch.reshape(labels, (labels_size[0] * labels_size[1], labels_size[2]))
        logits = model(inputs)
        # logits = outputs.squeeze()

        # pred_out = torch.sigmoid(logits)
        # pred_out = pred_out.cpu().numpy()
        pred_out_sigmoid = torch.sigmoid(logits).cpu().numpy()
        pred_out_int = np.rint(pred_out_sigmoid)
        labels = np.array(labels.cpu())
        ###binary
        pred_out_binary = np.rint(pred_out_int.ravel().tolist())
        labels_binary = labels.ravel().tolist()
        pred_list_binary.extend(pred_out_binary)
        label_list_binary.extend(labels_binary)
        ##multi-binary
        label_list.extend(labels)
        pred_int_list.extend(pred_out_int)
        pred_sigmoid_list.extend(pred_out_sigmoid)
        # except:
        #     print(inputs.size())
        #     print(batch['labels'].size())
    evaluation(label_list, pred_int_list, pred_sigmoid_list, label_list_binary, pred_list_binary, epoch_loss_avg)


def preder(model, tstloader, cfg):
    print(">>>>INFER IS PRED")
    pred_list = []
    index_list = []
    for step, batch in enumerate(tstloader):
        inputs = batch["image"]
        inputs = inputs.to(cfg.device, dtype=torch.float)
        logits = model(inputs)
        pred_out = torch.sigmoid(logits)
        pred_out = pred_out.cpu().numpy()
        pred_out = pred_out.ravel().tolist()
        index_out = createIndex(batch['imgname'], cfg)
        pred_list.append(pred_out)
        index_list.append(index_out)
    pred_list = np.concatenate(pred_list, 0)
    index_list = np.concatenate(index_list, 0)
    pred_dict = {'ID': index_list, 'Label': pred_list}
    pred_dict_df = pd.DataFrame(pred_dict)
    pred_dict_df.to_csv(
        '../predcsv/model{}_size{}_fold{}_ep{}.csv.gz'.format(cfg.model, cfg.size, cfg.fold, cfg.load_start),
        index=False)
