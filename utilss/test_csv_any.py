import numpy as np
import sklearn.metrics
import pandas as pd
import re

def evaluation_main(label_gt_list,label_predict_list,type = 'any'):
    # f1_score
    print("the f1_score of {} is {}".format(type,sklearn.metrics.f1_score(label_gt_list, label_predict_list)))#average='weighted'
    # recall_score
    print("recall_score of {} is {}".format(type,
        sklearn.metrics.recall_score(label_gt_list, label_predict_list)))
    # r2_score
    # print('r2_score of {} is {}'.format(type,sklearn.metrics.r2_score(label_gt_list, label_predict_list)))
    # accuracy_score
    print('accuracy_score of {} is {}'.format(type,sklearn.metrics.accuracy_score(label_gt_list, label_predict_list)))
    # precision_score
    print('precision_score of {} is {}'.format(type,
        sklearn.metrics.precision_score(label_gt_list, label_predict_list)))
    # average_precision_score
    # print('average_precision_score of {} is {}'.format(type,
    #     sklearn.metrics.average_precision_score(label_gt_list, label_predict_list)))
def evaluation_save(label_gt_list,label_predict_list,type,lossfun,epoch_loss,epoch,model_para,time,size):
    para_log = "model_{}_loss_{}".format(model_para, lossfun)
    f = open("/media/ps/_data/ICH/ich-master/logs/" + para_log + "_" + "{}".format(size)+"_"+time+"_"+type, "a")
    print(">>>>model is {}".format(epoch), file=f)
    print('>>>>epoch is {}'.format(model_para), file=f)
    print("epoch{} loss:{}".format(epoch, epoch_loss), file=f)
    # label_predict_list = [int(item > 0.8) for item in label_predict_list]
    # label_gt_list = [int(item > 0.6) for item in label_gt_list]
    label_predict_list = np.rint(label_predict_list)
    label_gt_list = np.rint(label_gt_list)
    # f1_score
    print("the f1_score of {} is {}".format(type,sklearn.metrics.f1_score(label_gt_list, label_predict_list)), file=f)#average='weighted'
    # recall_score
    print("recall_score of {} is {}".format(type,
        sklearn.metrics.recall_score(label_gt_list, label_predict_list)), file=f)
    # r2_score
    # print('r2_score of {} is {}'.format(type,sklearn.metrics.r2_score(label_gt_list, label_predict_list)))
    # accuracy_score
    print('accuracy_score of {} is {}'.format(type,sklearn.metrics.accuracy_score(label_gt_list, label_predict_list)), file=f)
    # precision_score
    print('precision_score of {} is {}'.format(type,
        sklearn.metrics.precision_score(label_gt_list, label_predict_list)), file=f)
    # average_precision_score
    # print('average_precision_score of {} is {}'.format(type,
    #     sklearn.metrics.average_precision_score(label_gt_list, label_predict_list)))
def test_sick(predict_csv,trn_sick,lossfun,epoch_loss,epoch,model_para,time):
    label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    sick = label_cols[trn_sick]
    gt_csv = pd.read_csv("/media/ps/_data/ICH/rsna-master/data/test_submission.csv", index_col=0)
    gt_csv = gt_csv.reindex(sorted(gt_csv.index)).reset_index()
    predict_csv = predict_csv.reindex(sorted(predict_csv.index)).reset_index()
    predict_csv = predict_csv[predict_csv['ID'].str.contains(sick)]
    label_predict_list = predict_csv['Label'].values.tolist()
    label_gt_list = gt_csv['Label'].values.tolist()
    evaluation_save(label_gt_list, label_predict_list, sick,lossfun,epoch_loss,epoch,model_para,time)
def test_case(label_list, pred_int_list,pred_sigmoid_list):
    # print("len(label_list)/32",len(label_list)/32)
    countcase = int(len(label_list)/32)
    case_pred_list = []
    case_label_list = []
    case_logit_list = []
    # print(pred_int_list[20])
    # print(pred_int_list[20][1])

    for i in range(countcase):
        case_pred = np.array([0, 0, 0, 0, 0, 0])
        case_label = np.array([0, 0, 0, 0, 0, 0])
        case_logit = np.array([0, 0, 0, 0, 0, 0])
        max = np.array([0.00000001, 0.00000001, 0.00000001, 0.00000001, 0.00000001, 0.00000001])
        for j in range(32):
            for k in range(6):
                if pred_int_list[j * i][k] == 1:
                    case_pred[k] = 1
                if label_list[j * i][k] == 1:
                    case_label[k] = 1
                if pred_sigmoid_list[j * i][k] > max[k]:
                    # max[k] = 2
                    max[k] = pred_sigmoid_list[j * i][k]
                    # print('pred_sigmoid_list[j * i][k]',pred_sigmoid_list[j * i][k])
                    # print("max[k-1]",max[k])
        case_logit = max
        case_pred_list.append(case_pred)
        case_label_list.append(case_label)
        case_logit_list.append(case_logit)
    return  case_label_list,case_pred_list,case_logit_list

                # if label_list[j * i][k] == 1:
                #     case_label[k] = 1

def test_sick_any_nolabel(label_list_binary, pred_list_binary):
    healthyList = []
    epiduralList = []
    intraparenchymalList = []
    intraventricularList = []
    subarachnoidList = []
    subduralList = []
    healthyList_label = []
    epiduralList_label = []
    intraparenchymalList_label = []
    intraventricularList_label = []
    subarachnoidList_label = []
    subduralList_label = []
    i = 0
    while i < len(label_list_binary):
        healthyList.append(pred_list_binary[i])
        epiduralList.append(pred_list_binary[i+1])
        intraparenchymalList.append(pred_list_binary[i+2])
        intraventricularList.append(pred_list_binary[i+3])
        subarachnoidList.append(pred_list_binary[i+4])
        subduralList.append(pred_list_binary[i+5])
        healthyList_label.append(label_list_binary[i])
        epiduralList_label.append(label_list_binary[i+1])
        intraparenchymalList_label.append(label_list_binary[i+2])
        intraventricularList_label.append(label_list_binary[i+3])
        subarachnoidList_label.append(label_list_binary[i+4])
        subduralList_label.append(label_list_binary[i+5])
        i = i + 6
    evaluation_main(healthyList_label,healthyList,'healthy')
    evaluation_main(epiduralList_label, epiduralList, 'epidural')
    evaluation_main(intraparenchymalList_label, intraparenchymalList, 'intraparenchymal')
    evaluation_main(intraventricularList_label, intraventricularList, 'ntraventricular')
    evaluation_main(subarachnoidList_label, subarachnoidList, 'subarachnoid')
    evaluation_main(subduralList_label, subduralList, 'subdural')
def test_sick_any(predict_csv,trn_sick,lossfun,epoch_loss,epoch,model_para,time):
    pattern = re.compile(r'.*_epidural$')
    label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    sick = label_cols[trn_sick]
    gt_csv = pd.read_csv("/media/ps/_data/ICH/rsna-master/data/test_submission.csv", index_col=0)
    gt_csv = gt_csv[gt_csv['ID'].str.contains(sick)].reset_index(drop=True)
    predict_csv = predict_csv[predict_csv['ID'].str.contains(sick)].reset_index(drop=True)
    gt_csv = gt_csv.set_index('ID')
    predict_csv = predict_csv.set_index('ID')
    predict_csv = predict_csv.reindex(sorted(predict_csv.index)).reset_index()
    gt_csv = gt_csv.reindex(sorted(gt_csv.index)).reset_index()
    label_predict_list = predict_csv['Label'].values.tolist()
    label_gt_list = gt_csv['Label'].values.tolist()
    evaluation_save(label_gt_list, label_predict_list, sick,lossfun,epoch_loss,epoch,model_para,time)
# def test_val(predict_csv,gt_csv,trn_sick,lossfun,epoch_loss,epoch,model_para,time):
#     evaluation_save(gt_csv, predict_csv, sick, lossfun, epoch_loss, epoch, model_para, time)
if __name__ == '__main__':
    predict_csv = pd.read_csv("../predcsv/modelresnest101_size224_fold1_ep17.csv.gz")
    gt_csv = pd.read_csv("/media/ps/_data/ICH/rsna-master/data/test_submission.csv", index_col=0)
    gt_csv = gt_csv.set_index('ID')
    predict_csv = predict_csv.set_index('ID')
    predict_csv = predict_csv.reindex(sorted(predict_csv.index)).reset_index()
    gt_csv = gt_csv.reindex(sorted(gt_csv.index)).reset_index()
    predict_csv = predict_csv.reindex(sorted(predict_csv.index)).reset_index()
    gt_csv = gt_csv.reindex(sorted(gt_csv.index)).reset_index()
    predict_csv_epidural = predict_csv[predict_csv['ID'].str.contains('epidural')]
    gt_csv_epidural = gt_csv[gt_csv['ID'].str.contains('epidural')]
    predict_csv_intraparenchymal = predict_csv[predict_csv['ID'].str.contains('intraparenchymal')]
    gt_csv_intraparenchymal = gt_csv[gt_csv['ID'].str.contains('intraparenchymal')]
    predict_csv_intraventricular = predict_csv[predict_csv['ID'].str.contains('intraventricular')]
    gt_csv_intraventricular = gt_csv[gt_csv['ID'].str.contains('intraventricular')]
    predict_csv_subarachnoid = predict_csv[predict_csv['ID'].str.contains('subarachnoid')]
    gt_csv_subarachnoid = gt_csv[gt_csv['ID'].str.contains('subarachnoid')]
    predict_csv_subdural = predict_csv[predict_csv['ID'].str.contains('subdural')]
    gt_csv_subdural = gt_csv[gt_csv['ID'].str.contains('subdural')]

    label_predict_list_epidural = predict_csv_epidural['Label'].values.tolist()
    label_gt_list_epidural = gt_csv_epidural['Label'].values.tolist()
    label_predict_list_intraparenchymal = predict_csv_intraparenchymal['Label'].values.tolist()
    label_gt_list_intraparenchymal = gt_csv_intraparenchymal['Label'].values.tolist()
    label_predict_list_intraventricular = predict_csv_intraventricular['Label'].values.tolist()
    label_gt_list_intraventricular = gt_csv_intraventricular['Label'].values.tolist()
    label_predict_list_subarachnoidl = predict_csv_subarachnoid['Label'].values.tolist()
    label_gt_list_subarachnoidl = gt_csv_subarachnoid['Label'].values.tolist()
    label_predict_list_subdural = predict_csv_subdural['Label'].values.tolist()
    label_gt_list_subdural = gt_csv_subdural['Label'].values.tolist()

    label_predict_list_epidural = np.rint(label_predict_list_epidural)
    label_predict_list_intraparenchymal = np.rint(label_predict_list_intraparenchymal)
    label_predict_list_intraventricular = np.rint(label_predict_list_intraventricular)
    label_predict_list_subarachnoidl = np.rint(label_predict_list_subarachnoidl)
    label_predict_list_subdural = np.rint(label_predict_list_subdural)
    # label_gt_list = np.rint(label_gt_list)
    evaluation_main(label_gt_list_epidural,label_predict_list_epidural,'epidural')
    evaluation_main(label_gt_list_intraparenchymal, label_predict_list_intraparenchymal,'intraparenchymal')
    evaluation_main(label_gt_list_intraventricular, label_predict_list_intraventricular,'intraventricular')
    evaluation_main(label_gt_list_subarachnoidl, label_predict_list_subarachnoidl,'subarachnoidl')
    evaluation_main(label_gt_list_subdural, label_predict_list_subdural,'subdural')