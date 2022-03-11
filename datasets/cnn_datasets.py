from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
from torch.utils.data import Dataset
import cv2
from sampler import multilabelClassSampler
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from albumentations import (Compose, Normalize, HorizontalFlip,
                            ShiftScaleRotate, Transpose
                            )
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """

    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    # logger.info(image.shape)
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype='uint8')
    imageout[:image.shape[0], :image.shape[1], :] = image.copy()
    return imageout


# def collatefn(batch):
#     maxlen = max([l['emb'].shape[0] for l in batch])
#     embdim = batch[0]['emb'].shape[1]
#     withlabel = 'labels' in batch[0]
#     if withlabel:
#         labdim = batch[0]['labels'].shape[1]
#         # maxlen = max([l['emb'].shape[0] for l in batch]) - 5
#     for b in batch:
#         masklen = maxlen - len(b['emb'])
#         b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
#         b['embidx'] = torch.cat((torch.ones((masklen), dtype=torch.long) * -1, b['embidx']))
#         b['mask'] = np.ones((maxlen))
#         b['mask'][:masklen] = 0.
#         if withlabel:
#             b['labels'] = np.vstack((np.zeros((maxlen - len(b['labels']), labdim)), b['labels']))
#     outbatch = {'emb': torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
#                                                for b in batch])).float()}
#     outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
#                                                for b in batch])).float()
#     outbatch['embidx'] = torch.tensor(np.vstack([np.expand_dims(b['embidx'], 0) \
#                                                  for b in batch])).float()
#     if withlabel:
#         outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
#     return outbatch
class IntracranialDataset(Dataset):
    def __init__(self, df, path, labels, transform=None, AUTOCROP=True, SIZE=224, label_cols=True, trn_sick=0,
                 type=None):
        self.data = df
        self.path = path
        self.patients = df.PatientID.unique()
        self.transform = transform
        self.labels = labels
        self.crop = AUTOCROP
        self.size = SIZE
        self.label_cols = label_cols
        self.trn_sick = trn_sick
        self.type = type
        # if self.type == 'train':
        #     self.data = self.data.set_index('PatientID')
        # if self.type == 'test':
        #     self.data = self.data.set_index('PatientID')#df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.jpg')
        img = cv2.imread(img_name)
        if self.crop:
            try:
                try:
                    img = autocrop(img, threshold=0, kernsel_size=img.shape[0] // 15)
                except:
                    img = autocrop(img, threshold=0)
            except:
                1
        img = cv2.resize(img, (self.size, self.size))
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        if self.labels:
            labels = torch.tensor(
                [self.data.loc[idx, self.label_cols]])
        return {'image': img, 'labels': labels}

    def insert_df(self, df, location):
        # print("location",location)
        # print("size",df.shape)
        df_head = df.iloc[:location]
        df_tail = df.iloc[location:]
        slices = df.iloc[[location]]
        df = pd.concat([df_head, slices])
        df = pd.concat([df, df_tail])
        return df

    def getTrainbatch(self, patdf):
        patdf = patdf.reset_index()
        shape = patdf.shape[0]
        if shape > 32:
            if shape % 2 != 0:
                patdf = patdf[:shape - 2]
                shape = shape - 1
            if shape != 32:
                diff = shape - 32
                delete = int(diff / 2)
                patdf = patdf[delete:shape - delete]
                # print("1111/size",patdf.shape)
        if shape < 32:  # 思路插入一些值，然后做mixup，减少loss的权重
            if shape % 2 != 0:
                patdf = patdf[:shape - 1]
                shape = shape - 1
            # print('shape',shape)
            # print("2222/size", patdf.shape)
            diff = 32 - shape
            insert_location = int(shape / 2)
            insert_count = int(diff / 2)
            count = 0
            # patdf = patdf[insert_location-insert_count:insert_location+insert_count]

            for i in range(diff):
                location = i + insert_location - insert_count + count
                patdf = self.insert_df(patdf, location)
                count = count + 1
            # print("3333/size", patdf.shape)
        if patdf.shape[0] == 31:
            patdf = self.insert_df(patdf, 15)
            # print(patdf.shape[0])
        # print("sad",patdf.shape[0])
        # print("4444/size", patdf.shape)
        imagesList = []
        labelsList = []
        for idx, slice in patdf.iterrows():
            img_name = os.path.join(self.path, slice['Image'] + '.jpg')
            # img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(img_name)
            if self.crop:
                try:
                    try:
                        img = autocrop(img, threshold=0, kernsel_size=img.shape[0] // 15)
                    except:
                        img = autocrop(img, threshold=0)
                except:
                    1
            # print("img",img)
            img = cv2.resize(img, (self.size, self.size))
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            # if self.labels:
            # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'
            labels = torch.tensor(
                [slice['healthy'], slice['epidural'], slice['intraparenchymal'], slice['intraventricular'],
                 slice['subarachnoid'], slice['subdural']])
            imagesList.append(img.numpy())
            labelsList.append(labels.numpy())
            # dict = {'image': img, 'labels': labels}
            # dict_batch.append(dict)
        images = torch.tensor(imagesList)
        labelss = torch.tensor(labelsList)
        # images = torch.tensor(item.numpy() for item in imagesList)
        # labelss = torch.tensor(item.numpy() for item in labelsList)
        dict = {'image': images, 'labels': labelss}
        return dict


def datasets(trndf, valdf, testdf, cfg):
    path_data = cfg.dataset
    # 这里改图片路径
    dir_train_img = os.path.join(path_data, 'train_no_brain/')
    dir_test_img = os.path.join(path_data, 'test_no_brain/')

    print('Trn shape {} {}'.format(*trndf.shape))
    print('Val shape {} {}'.format(*valdf.shape))
    # Data loaders
    mean_img = [0.22363983, 0.18190407, 0.2523437]
    std_img = [0.32451536, 0.2956294, 0.31335256]
    transform_train = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                         rotate_limit=20, p=0.3, border_mode=cv2.BORDER_REPLICATE),
        Transpose(p=0.5),
        Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])
    HFLIPVAL = 0.2 if cfg.hflip == True else 0.0
    TRANSPOSEVAL = 0.2 if cfg.transpose == True else 0.0
    transform_test = Compose([
        # HorizontalFlip(p=HFLIPVAL),
        # Transpose(p=TRANSPOSEVAL),
        Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])
    # no val
    trndataset = IntracranialDataset(trndf, path=dir_train_img, transform=transform_train, labels=True,
                                     SIZE=cfg.size, label_cols=cfg.label_cols, AUTOCROP=cfg.autocrop, type='train')
    valdataset = IntracranialDataset(valdf, path=dir_train_img, transform=transform_test, labels=True,
                                     SIZE=cfg.size, label_cols=cfg.label_cols, AUTOCROP=cfg.autocrop, type='test')
    tstdataset = IntracranialDataset(testdf, path=dir_test_img, transform=transform_test, labels=True,
                                     SIZE=cfg.size, label_cols=cfg.label_cols, AUTOCROP=cfg.autocrop, type='test')
    num_workers = 16
    traindf = trndataset.data
    t = traindf[['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']]
    labels = t.values
    # least_sampled = multilabelClassSampler.MultilabelBalancedRandomSampler(labels,indices=None, class_choice="least_sampled")
    trnloader = DataLoader(trndataset, batch_size=cfg.batchsize, num_workers=num_workers, shuffle=True,
                           pin_memory=True)  # sampler=least_sampled,collate_fn = collatefn
    valloader = DataLoader(valdataset, batch_size=cfg.batchsize, shuffle=False, num_workers=num_workers,
                           pin_memory=True)
    tstloader = DataLoader(tstdataset, batch_size=cfg.batchsize, shuffle=False, num_workers=num_workers,
                           pin_memory=True)
    return trnloader, valloader, tstloader


def datasets_emb(trndf, valdf, testdf, cfg):
    path_data = cfg.dataset
    # 这里改图片路径
    dir_train_img = os.path.join(path_data, 'train_no_brain/')
    dir_test_img = os.path.join(path_data, 'test_no_brain/')

    print('Trn shape {} {}'.format(*trndf.shape))
    print('Val shape {} {}'.format(*valdf.shape))
    # Data loaders
    mean_img = [0.22363983, 0.18190407, 0.2523437]
    std_img = [0.32451536, 0.2956294, 0.31335256]
    transform_train = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                         rotate_limit=20, p=0.3, border_mode=cv2.BORDER_REPLICATE),
        Transpose(p=0.5),
        Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])
    HFLIPVAL = 0.2 if cfg.hflip == True else 0.0
    TRANSPOSEVAL = 0.2 if cfg.transpose == True else 0.0
    transform_test = Compose([
        # HorizontalFlip(p=HFLIPVAL),
        # Transpose(p=TRANSPOSEVAL),
        Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])
    # no val
    trndataset = IntracranialDataset(trndf, path=dir_train_img, transform=transform_train, labels=True,
                                     SIZE=cfg.size, label_cols=cfg.label_cols, AUTOCROP=cfg.autocrop, type='train')
    valdataset = IntracranialDataset(valdf, path=dir_train_img, transform=transform_test, labels=True,
                                     SIZE=cfg.size, label_cols=cfg.label_cols, AUTOCROP=cfg.autocrop, type='test')
    tstdataset = IntracranialDataset(testdf, path=dir_test_img, transform=transform_test, labels=True,
                                     SIZE=cfg.size, label_cols=cfg.label_cols, AUTOCROP=cfg.autocrop, type='test')
    num_workers = 16
    traindf = trndataset.data
    t = traindf[['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']]
    labels = t.values
    least_sampled = multilabelClassSampler.MultilabelBalancedRandomSampler(labels, indices=None,
                                                                           class_choice="least_sampled")
    trnloader = DataLoader(trndataset, batch_size=cfg.test_batchsize, num_workers=num_workers,
                           shuffle=True)  # sampler=least_sampled,collate_fn = collatefn
    valloader = DataLoader(valdataset, batch_size=cfg.test_batchsize, shuffle=False, num_workers=num_workers)
    tstloader = DataLoader(tstdataset, batch_size=cfg.test_batchsize, shuffle=False, num_workers=num_workers)
    return trnloader, valloader, tstloader


def datasets_test(testdf, dir_test_img, cfg):
    # Data loaders
    mean_img = [0.22363983, 0.18190407, 0.2523437]
    std_img = [0.32451536, 0.2956294, 0.31335256]
    transform_test = Compose([
        # HorizontalFlip(p=HFLIPVAL),
        # Transpose(p=TRANSPOSEVAL),
        Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])
    # no val
    tstdataset = IntracranialDataset(testdf, path=dir_test_img, transform=transform_test, labels=True,
                                     SIZE=cfg.size, label_cols=cfg.label_cols, AUTOCROP=cfg.autocrop, type='test')
    num_workers = 16
    tstloader = DataLoader(tstdataset, batch_size=cfg.batchsize * 2, shuffle=False, num_workers=num_workers)
    return tstloader
