import os
from configs import config_cnn as cfg
# import configs.config_cnn as cfg
from trainer import tools
import torch
from torch.backends import cudnn
from tqdm import tqdm
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
import wandb
from model import resnet
# wandb.init(project='ich')
import shutil
warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True


def insert_df(df, location):
    # print("location",location)
    # print("size",df.shape)
    df_head = df.iloc[:location]
    df_tail = df.iloc[location:]
    slices = df.iloc[[location]]
    df = pd.concat([df_head, slices])
    df = pd.concat([df, df_tail])
    return df
if __name__ == '__main__':
    path_data = cfg.dataset
    dir_train_img = os.path.join(path_data, 'dataset/train/')
    dir_test_img = os.path.join(path_data, 'dataset/test/')
    train = pd.read_csv(os.path.join(path_data, 'mytrain_new.csv'))
    testdf = pd.read_csv(os.path.join(path_data, 'mytest_new.csv'))

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
    # valdf = train[train['fold']== cfg.fold].reset_index(drop=True)
    trndf = train[train['fold']!= 4].reset_index(drop=True)#fold 2
    print('Trn shape {} {}'.format(*trndf.shape))
    print('Tst shape {} {}'.format(*testdf.shape))
    # for debug
    # trndf = train[:2000]
    # testdf = testdf[:2000]
    # valdf = valdf[:2000]

    #seq
    trnmdf = pd.read_csv(os.path.join(path_data, 'new_train_metadata.csv'))
    tstmdf = pd.read_csv(os.path.join(path_data, 'new_test_metadata.csv'))
    trnmdf['SliceID'] = trnmdf[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(
        lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
    tstmdf['SliceID'] = tstmdf[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(
        lambda x: '{}__{}__{}'.format(*x.tolist()), 1)

    poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
    trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient'] \
                                   .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
    tstmdf[poscols] = pd.DataFrame(tstmdf['ImagePositionPatient'] \
                                   .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())

    trnmdf = trnmdf.sort_values(['SliceID'] + poscols) \
        [['PatientID', 'SliceID', 'SOPInstanceUID'] + poscols].reset_index(drop=True)
    tstmdf = tstmdf.sort_values(['SliceID'] + poscols) \
        [['PatientID', 'SliceID', 'SOPInstanceUID'] + poscols].reset_index(drop=True)

    trnmdf['seq'] = (trnmdf.groupby(['SliceID']).cumcount() + 1)
    tstmdf['seq'] = (tstmdf.groupby(['SliceID']).cumcount() + 1)

    keepcols = ['PatientID', 'SliceID', 'SOPInstanceUID', 'seq']
    trnmdf = trnmdf[keepcols]  #  PattientID SliceId instanceID seq
    tstmdf = tstmdf[keepcols]  #  PattientID SliceId instanceID seq
    trnmdf.columns = tstmdf.columns = ['PatientID', 'SliceID', 'Image', 'seq']
    trnmdf = trnmdf.drop('PatientID',axis=1)
    trnmdf = trnmdf.drop('SliceID',axis=1)
    trndf_addseq = pd.merge(trndf,trnmdf,on= 'Image')
    tstmdf = tstmdf.drop('PatientID', axis=1)
    tstmdf = tstmdf.drop('SliceID', axis=1)
    tstmdf_addseq = pd.merge(testdf, tstmdf, on='Image')
    ##end seq
    trn_patients = trndf_addseq.PatientID.unique()
    tst_patients = tstmdf_addseq.PatientID.unique()
    patient_trndf = trndf_addseq.set_index('PatientID')
    patient_tstdf = tstmdf_addseq.set_index('PatientID')
    new_patient_trndf = patient_trndf.copy()
    new_patient_trndf.drop(new_patient_trndf.index,inplace=True)
    new_patient_tstdf = patient_tstdf.copy()
    new_patient_tstdf.drop(new_patient_tstdf.index, inplace=True)
    patient_trndf_len = len(trn_patients)
    patient_tstdf_len = len(tst_patients)
    # dict = {'Image':imagelist,'PatientID':patientlist,'healthy':healthylist,'epidural':epidurallist,'intraparenchymal':intraparenchymallist,'Image':imagelist,}
    for patidx in tqdm(range(patient_trndf_len-1)):
        patidx = trn_patients[patidx]
        patdf_init = patient_trndf.loc[patidx].sort_values('seq')
        patdf_init = patdf_init.drop_duplicates(['seq'])
        patdf_init = patdf_init.loc[patidx].sort_values('seq')
        patdf = patdf_init.reset_index()
        shape = patdf.shape[0]
        if shape<14:
            if shape % 2 != 0:
                patdf = patdf[:shape - 1]
                shape = shape - 1
                # print('shape',shape)
                # print("2222/size", patdf.shape)
            diff = 14 - shape
            insert_location = int(shape / 2)
            insert_count = int(diff / 2)
            count = 0
            # patdf = patdf[insert_location-insert_count:insert_location+insert_count]

            for i in range(diff):
                location = i + insert_location - insert_count + count
                patdf = insert_df(patdf, location)
                count = count + 1
        else:
            if shape % 2 != 0:
                patdf = patdf[1:]
                shape = shape - 1
                # print('shape',shape)
                # print("2222/size", patdf.shape)
            diff = 14 - shape
            if shape != 14:
                diff = shape - 14
                delete= int(diff/2)
                patdf = patdf[delete:shape-delete]
        patdf = patdf.set_index('PatientID')
        new_patient_trndf = pd.concat([new_patient_trndf,patdf])
    for patidx in tqdm(range(patient_tstdf_len-1)):
        patidx = tst_patients[patidx]
        patdf_init = patient_tstdf.loc[patidx].sort_values('seq')
        patdf_init = patdf_init.drop_duplicates(['seq'])
        patdf_init = patdf_init.loc[patidx].sort_values('seq')
        patdf = patdf_init.reset_index()
        shape = patdf.shape[0]
        if shape<14:
            if shape % 2 != 0:
                patdf = patdf[:shape - 1]
                shape = shape - 1
                # print('shape',shape)
                # print("2222/size", patdf.shape)
            diff = 14 - shape
            insert_location = int(shape / 2)
            insert_count = int(diff / 2)
            count = 0
            # patdf = patdf[insert_location-insert_count:insert_location+insert_count]

            for i in range(diff):
                location = i + insert_location - insert_count + count
                patdf = insert_df(patdf, location)
                count = count + 1
        else:
            if shape % 2 != 0:
                patdf = patdf[1:]
                shape = shape - 1
                # print('shape',shape)
                # print("2222/size", patdf.shape)
            diff = 14 - shape
            if shape != 14:
                diff = shape - 14
                delete= int(diff/2)
                patdf = patdf[delete:shape-delete]
        patdf = patdf.set_index('PatientID')
        new_patient_tstdf = pd.concat([new_patient_tstdf,patdf])
    new_patient_tstdf  = new_patient_tstdf.reset_index()
    new_patient_trndf = new_patient_trndf.reset_index()
    trn_image_list = new_patient_trndf['Image'].tolist()
    tst_image_list= new_patient_tstdf['Image'].tolist()
    trndata_path = '/media/ps/_data/ICH/rsna-master/data/dataset/train/'
    tstdata_path = '/media/ps/_data/ICH/rsna-master/data/dataset/test/'
    trn_to_path = '/media/ps/_data/ICH/ich-master/dataset/train/'
    tst_to_path = '/media/ps/_data/ICH/ich-master/dataset/test/'
    for img in tqdm(trn_image_list):
        imgname = img +'.jpg'
        copyname = os.path.join(trndata_path,imgname)
        toname = os.path.join(trn_to_path,imgname)
        shutil.copy(copyname,toname)
    for img in tqdm(tst_image_list):
        imgname = img + '.jpg'
        copyname = os.path.join(tstdata_path, imgname)
        toname = os.path.join(tst_to_path, imgname)
        shutil.copy(copyname, toname)
    #加.jpg,然后进行copy到新的数据集中
    new_patient_trndf.to_csv('/media/ps/_data/ICH/ich-master/dataset/newtrain.csv')
    new_patient_tstdf.to_csv('/media/ps/_data/ICH/ich-master/dataset/newtest.csv')