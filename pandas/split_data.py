import os
from configs import config_cnn as cfg
import pandas as pd
import glob
import numpy as np
import ast
from datasets import cnn_datasets
if __name__ == '__main__':

    trn_csv = pd.read_csv('/media/ps/_data/ICH/ich-master/dataset/newtrain.csv',index_col=0)
    tst_csv = pd.read_csv('/media/ps/_data/ICH/ich-master/dataset/newtest.csv',index_col=0)
    trndf_patient = trn_csv.set_index('PatientID')
    tstdf_patient = tst_csv.set_index('PatientID')
    trnindex = trndf_patient.index.unique()
    tstindex = tstdf_patient.index.unique()
    dict = {'PatientID':[],'healthy':[],'epidural':[],'intraparenchymal':[],'intraventricular':[],'subarachnoid':[],'subdural':[],'fold':[]}
    dict_tst = {'PatientID':[],'healthy':[],'epidural':[],'intraparenchymal':[],'intraventricular':[],'subarachnoid':[],'subdural':[]}
    #label_cols = ['healthy','epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']#, 'healthy'
    for index  in trnindex:
        patdf = trndf_patient.loc[index].sort_values('seq')
        dict['PatientID'].append(index)
        # print(index)
        for idx,row in patdf.iteritems():
            if idx == 'fold':
                a = patdf[idx].values[0]
                dict[idx].append(patdf[idx].values[0])
                continue
            # print(idx)
            if idx == "seq":
                continue
            if idx == "Image":
                continue
            if idx == 'healthy' :
                healthy = patdf[idx].values.tolist()
                if 0 in healthy:
                    dict[idx].append(0)
                else:
                    dict[idx].append(1)
            else:
                ich = patdf[idx].values.tolist()
                if 1 in ich:
                    dict[idx].append(1)
                else:
                    dict[idx].append(0)
    for index  in tstindex:
        patdf = tstdf_patient.loc[index].sort_values('seq')
        dict_tst['PatientID'].append(index)
        # print(index)
        for idx,row in patdf.iteritems():
            # print(idx)
            if idx == "fold":
                continue
            if idx == "seq":
                continue
            if idx == "Image":
                continue
            if idx == 'healthy' :
                healthy = patdf[idx].values.tolist()
                if 0 in healthy:
                    dict_tst[idx].append(0)
                else:
                    dict_tst[idx].append(1)
            else:
                ich = patdf[idx].values.tolist()
                if 1 in ich:
                    dict_tst[idx].append(1)
                else:
                    dict_tst[idx].append(0)
    label_df_trn = pd.DataFrame(dict)
    label_df_trn.to_csv('/media/ps/_data/ICH/ich-master/dataset/trn_patient_labels.csv',index=False)
    label_df_tst = pd.DataFrame(dict_tst)
    label_df_tst.to_csv('/media/ps/_data/ICH/ich-master/dataset/tst_patient_labels.csv',index=False)