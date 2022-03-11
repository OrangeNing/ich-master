import numpy as np
import sklearn.metrics
import pandas as pd
import os
import time
import datetime
# print(matplotlib.matplotlib_fname())
# print(matplotlib.get_cachedir())
# predict_csv = pd.read_csv("submission/submission.csv.gz")
# predict_csv = pd.read_csv("../../data/test_submission.csv")
def test(predict_csv,lossfun,epoch_loss,epoch,model_para,time):
    # time = d.time()
    para_log ="model_{}_loss_{}".format(model_para,lossfun)
    f = open("/media/ps/_data/ICH/rsna-master/logs/"+para_log+"_"+time, "a")
    print(">>>>model is {}".format(epoch),file = f)
    print('>>>>epoch is {}'.format(model_para),file = f)
    print("epoch{} loss:{}".format(epoch, epoch_loss), file=f)
    gt_csv = pd.read_csv("data/test_submission.csv", index_col = 0)
    gt_csv = gt_csv.set_index('ID')
    predict_csv = predict_csv.set_index('ID')
    predict_csv = predict_csv.reindex(sorted(predict_csv.index)).reset_index()
    gt_csv = gt_csv.reindex(sorted(gt_csv.index)).reset_index()
    label_predict_list = predict_csv['Label'].values.tolist()
    label_gt_list = gt_csv['Label'].values.tolist()
    #auc
    fpr,tpr,thresholds = sklearn.metrics.roc_curve(label_gt_list,label_predict_list)
    # print('fpr',fpr)
    # print('tpr',tpr)
    # print(thresholds)
    roc_auc = sklearn.metrics.auc(fpr,tpr)
    print("roc_auc is {}".format(roc_auc),file = f)
    #roc_score
    # plt.plot(fpr,tpr,lw =2,color = 'darkorange',label = 'ROC curve (area = %0.5f)' % roc_auc)
    # plt.plot([0,1],[0,1],color='navy',lw=2,linestyle = '--')
    # plt.xlabel('假正例率')
    # plt.ylabel('真正例率')
    # plt.legend(loc = 'lower right')
    # # plt.rcParams['font.sans-serif'] = [u'SimHei']
    # # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    # plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
    # plt.show()
    # label_predict_list = [int(item>0.1)for item in label_predict_list]
    label_predict_list = np.rint(label_predict_list)
    label_gt_list = np.rint(label_gt_list)
    ###
    #f1_score
    print("f1_score is {}".format(sklearn.metrics.f1_score(label_gt_list,label_predict_list)),file = f)
    #recall_score
    print("recall_score is {}".format(sklearn.metrics.recall_score(label_gt_list,label_predict_list)),file = f)
    #r2_score
    # print('r2_score is {}'.format(sklearn.metrics.r2_score(label_gt_list,label_predict_list)))
    #accuracy_score ,normalize = False)
    print('accuracy_score is {}'.format(sklearn.metrics.accuracy_score(label_gt_list,label_predict_list)),file = f)
    #precision_score
    print('precision_score is {}'.format(sklearn.metrics.precision_score(label_gt_list,label_predict_list)),file = f)
    f.close()
    #average_precision_score
    # print('average_precision_score is {}'.format(sklearn.metrics.average_precision_score(label_gt_list,label_predict_list)))
    #roc_auc_score
    # print('roc_auc_score is {}'.format(sklearn.metrics.roc_auc_score(label_gt_list,label_predict_list,average = 'binary')))
    ###

    # print("f1 is {}".format(evaluation.get_F1(label_predict_list,label_gt_list)))
    # time = d.time()
def test_main(predict_csv):
    # time = d.time()
    # para_log ="model_{}_loss_{}".format(model_para,lossfun)
    # print(">>>>model is {}".format(epoch),file = f)
    # print('>>>>epoch is {}'.format(model_para),file = f)
    # print("epoch{} loss:{}".format(epoch, epoch_loss), file=f)
    gt_csv = pd.read_csv("data/test_submission_new.csv", index_col = 0)
    # gt_csv.loc[:,~gt_csv.columns.str.contains("~Unnamed")]
    gt_csv = gt_csv.set_index('ID')
    predict_csv = predict_csv.set_index('ID')
    predict_csv = predict_csv.reindex(sorted(predict_csv.index)).reset_index()
    gt_csv = gt_csv.reindex(sorted(gt_csv.index)).reset_index()
    label_predict_list = predict_csv['Label'].values.tolist()
    label_gt_list = gt_csv['Label'].values.tolist()
    #auc
    fpr,tpr,thresholds = sklearn.metrics.roc_curve(label_gt_list,label_predict_list)
    # print('fpr',fpr)
    # print('tpr',tpr)
    # print(thresholds)
    roc_auc = sklearn.metrics.auc(fpr,tpr)
    print("roc_auc is {}".format(roc_auc))
    #roc_score
    # plt.plot(fpr,tpr,lw =2,color = 'darkorange',label = 'ROC curve (area = %0.5f)' % roc_auc)
    # plt.plot([0,1],[0,1],color='navy',lw=2,linestyle = '--')
    # plt.xlabel('假正例率')
    # plt.ylabel('真正例率')
    # plt.legend(loc = 'lower right')
    # # plt.rcParams['font.sans-serif'] = [u'SimHei']
    # # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    # plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
    # plt.show()
    # label_predict_list = [int(item>0.1)for item in label_predict_list]
    label_predict_list = np.rint(label_predict_list)
    label_gt_list = np.rint(label_gt_list)
    ###
    #f1_score
    print("f1_score is {}".format(sklearn.metrics.f1_score(label_gt_list,label_predict_list)))
    #recall_score
    print("recall_score is {}".format(sklearn.metrics.recall_score(label_gt_list,label_predict_list)))
    #r2_score
    # print('r2_score is {}'.format(sklearn.metrics.r2_score(label_gt_list,label_predict_list)))
    #accuracy_score ,normalize = False)
    print('accuracy_score is {}'.format(sklearn.metrics.accuracy_score(label_gt_list,label_predict_list)))
    #precision_score
    print('precision_score is {}'.format(sklearn.metrics.precision_score(label_gt_list,label_predict_list)))
if __name__ == '__main__':
    # print(os.getcwd())
    predict_csv = pd.read_csv("/media/ps/_data/ICH/rsna-master/predcsv/modelresnext101_augTP_size224_fold4_ep0.csv.gz")
    test_main(predict_csv)
    # print(datetime.datetime.now())