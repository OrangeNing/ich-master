from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pandas as pd
from configs import  config_lstm as lcfg
label_cols = lcfg.label_cols
class IntracranialDataset_pati(Dataset):
    def __init__(self, df,labels_df, mat, labels=True):
        self.data = df
        self.mat = mat
        self.labels = labels
        self.patients = df.PatientID.unique()
        self.data = self.data.set_index('PatientID')
        self.labels_df = labels_df.set_index('PatientID')

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patidx = self.patients[idx]
        patdf = self.data.loc[patidx].sort_values('seq')
        patdf = self.getTrainbatch(patdf)
        patdf = patdf.set_index('PatientID')
        patilabel_serices = self.labels_df.loc[patidx][1:]
        patemb = self.mat[patdf['embidx'].values]

        # patdeltalag = np.zeros(patemb.shape)
        # patdeltalead = np.zeros(patemb.shape)
        # patdeltalag[1:] = patemb[1:] - patemb[:-1]
        # patdeltalead[:-1] = patemb[:-1] - patemb[1:]
        #
        # patemb = np.concatenate((patemb, patdeltalag, patdeltalead), -1)

        ids = torch.tensor(patdf['embidx'].values)

        if self.labels:
            labels = torch.tensor(patdf[label_cols].values)
            patilabel = torch.tensor(patilabel_serices.values)
            return {'emb': patemb, 'embidx': ids, 'labels': labels,'patilabel':patilabel}
        else:
            return {'emb': patemb, 'embidx': ids}
    def insert_df(self,df,location):
        df_head = df.iloc[:location]
        df_tail = df.iloc[location:]
        slices = df.iloc[[location]]
        df = pd.concat([df_head,slices])
        df = pd.concat([df,df_tail])
        return df
    def getTrainbatch(self,patdf):
        patdf = patdf.reset_index()
        patdf = patdf.drop_duplicates(subset = ['seq'],keep = 'first')
        shape = patdf.shape[0]
        if shape>32:
            if shape %2 != 0 :
                patdf = patdf[:shape-2]
                shape = shape -1
            if shape != 32:
                diff = shape - 32
                delete= int(diff/2)
                patdf = patdf[delete:shape-delete]
                # print("1111/size",patdf.shape)
        if shape < 32:#思路插入一些值，然后做mixup，减少loss的权重
            if shape %2 != 0 :
                patdf = patdf[:shape-1]
                shape = shape - 1
            diff = 32 - shape
            insert_location = int(shape / 2)
            insert_count = int(diff / 2)
            count = 0
            # patdf = patdf[insert_location-insert_count:insert_location+insert_count]
            for i in range(diff):
                location = i + insert_location - insert_count + count
                patdf = self.insert_df(patdf, location)
                count = count + 1
        if patdf.shape[0] == 31:
            patdf = self.insert_df(patdf, 15)
        return patdf


class IntracranialDataset(Dataset):
    def __init__(self, df, mat, labels=label_cols):
        self.data = df
        self.mat = mat
        self.labels = labels
        self.patients = df.PatientID.unique()
        self.data = self.data.set_index('PatientID')

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        patidx = self.patients[idx]
        patdf = self.data.loc[patidx].sort_values('seq')
        patemb = self.mat[patdf['embidx'].values]

        # patdeltalag = np.zeros(patemb.shape)
        # patdeltalead = np.zeros(patemb.shape)
        # patdeltalag[1:] = patemb[1:] - patemb[:-1]
        # patdeltalead[:-1] = patemb[:-1] - patemb[1:]
        #
        # patemb = np.concatenate((patemb, patdeltalag, patdeltalead), -1)

        ids = torch.tensor(patdf['embidx'].values)

        if self.labels:
            labels = torch.tensor(patdf[label_cols].values)
            return {'emb': patemb, 'embidx': ids, 'labels': labels}
        else:
            return {'emb': patemb, 'embidx': ids}


def collatefn(batch):
    maxlen = max([l['emb'].shape[0] for l in batch])
    embdim = batch[0]['emb'].shape[1]
    withlabel = 'labels' in batch[0]
    if withlabel:
        labdim = batch[0]['labels'].shape[1]

    for b in batch:
        masklen = maxlen - len(b['emb'])
        b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
        b['embidx'] = torch.cat((torch.ones((masklen), dtype=torch.long) * -1, b['embidx']))
        b['mask'] = np.ones((maxlen))
        b['mask'][:masklen] = 0.
        if withlabel:
            b['labels'] = np.vstack((np.zeros((maxlen - len(b['labels']), labdim)), b['labels']))

    outbatch = {'emb': torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
                                               for b in batch])).float()}
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
                                               for b in batch])).float()
    outbatch['embidx'] = torch.tensor(np.vstack([np.expand_dims(b['embidx'], 0) \
                                                 for b in batch])).float()
    if withlabel:
        outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    return outbatch
def collatefn_old(batch):
    # maxlen = max([l['emb'].shape[0] for l in batch])
    # embdim = batch[0]['emb'].shape[1]
    # withlabel = 'labels' in batch[0]
    # if withlabel:
    #     labdim = batch[0]['labels'].shape[1]
    # for b in batch:
    #     # if withlabel:
    #     #     len_b = len(b['emb'])
    #     #     delete_list = [0, 1, 2, len_b - 1, len_b - 2]
    #     #     b['emb'] = np.delete(b['emb'], delete_list, axis=0)  # delete emb
    #     #     b['embidx'] = tensorCut1D(b['embidx'], 3, len_b - 2)
    #     #     b['labels'] = tensorCut2D(b['labels'], delete_list, 0)
    #     #     len_b = len_b - 5
    #     #     masklen = maxlen - len_b
    #     #     b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
    #     #     b['embidx'] = torch.cat((torch.ones((masklen), dtype=torch.long) * -1, b['embidx']))
    #     #     b['mask'] = np.ones((maxlen))
    #     #     b['mask'][:masklen] = 0.
    #     #     b['labels'] = np.vstack((np.zeros((maxlen - len(b['labels']), labdim)), b['labels']))
    #     # else:
    #     masklen = maxlen - len(b['emb'])
    #     b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
    #     b['embidx'] = torch.cat((torch.ones((masklen), dtype=torch.long) * -1, b['embidx']))
    #     b['mask'] = np.ones((maxlen))
    #     b['mask'][:masklen] = 0.
    #     if withlabel:
    #         b['labels'] = np.vstack((np.zeros((maxlen - len(b['labels']), labdim)), b['labels']))

    outbatch = {'emb': torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
                                               for b in batch])).float()}
    # outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
    #                                            for b in batch])).float()
    outbatch['embidx'] = torch.tensor(np.vstack([np.expand_dims(b['embidx'], 0) \
                                                 for b in batch])).float()
    outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    outbatch['patilabel'] = torch.tensor(np.vstack([np.expand_dims(b['patilabel'], 0) for b in batch])).float()
    return outbatch