import os
import datetime
import pandas as pd
import numpy as np
import cv2
import pydicom
from tqdm import tqdm
# from joblib import delayed, Parallel
import sys

sys.path.insert(0, 'scripts')
from utilss.logs import get_logger

# Print info about environments
logger = get_logger('Prepare Data', 'INFO')  # noqa
logger.info('Cuda set up : time {}'.format(datetime.datetime.now().time()))


def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {attr: cast(getattr(dicom, attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}


def rescale_image(image, slope, intercept):
    return image * slope + intercept


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def get_dicom_meta(dicom):
    return {
        'PatientID': dicom.PatientID,  # can be grouped (20-548)
        'StudyInstanceUID': dicom.StudyInstanceUID,  # can be grouped (20-60)
        'SeriesInstanceUID': dicom.SeriesInstanceUID,  # can be grouped (20-60)
        'WindowWidth': get_dicom_value(dicom.WindowWidth),
        'WindowCenter': get_dicom_value(dicom.WindowCenter),
        'RescaleIntercept': float(dicom.RescaleIntercept),
        'RescaleSlope': float(dicom.RescaleSlope),  # all same (1.0)
    }
# def create_jpg(image,name):
#     image -= image.min((0, 1))
#     image = (255 * image).astype(np.uint8)
#     cv2.imwrite(os.path.join(PATHPROC, name) + '.jpg', image)


def apply_window_policy(image):
    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    # image1 = apply_window(image, 40, 80) # brain
    # image2 = apply_window(image, 80, 200) # subdural
    # image3 = apply_window(image, 40, 380) # bone
    # image1 = (image1 - 0) / 80
    # image2 = (image2 - (-20)) / 130
    # image3 = (image3 - (-150)) / 2800

    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1, 2, 0)

    return image
def apply_window_policy_test(image):
    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 60, 2000) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    # image1 = apply_window(image, 40, 80) # brain
    # image2 = apply_window(image, 80, 200) # subdural
    # image3 = apply_window(image, 40, 380) # bone
    # image1 = (image1 - 0) / 80
    # image2 = (image2 - (-20)) / 130
    # image3 = (image3 - (-150)) / 2800

    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1, 2, 0)

    return image3 - image3.mean()

def convert_dicom_to_jpg(name):
    try:
        data = os.path.join('/media/ps/_data/ICH/rsna-master/data/proc', name)
        # dirtype = 'train' if 'train' in name else 'test'
        imgnm = (name.split('/')[-1]).replace('.dcm', '')
        dicom = pydicom.dcmread(data)#DicomBytesIO(data)
        image = dicom.pixel_array
        image = rescale_image(image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
        # cv2.imwrite(os.path.join(PATHPROC, imgnm) + 'raw.jpg', image)
        # image = apply_window_policy(image)
        image = apply_window_policy_test(image)
        image -= image.min((0, 1))
        image = (255 * image).astype(np.uint8)
        cv2.imwrite(os.path.join(PATHPROC, imgnm) + 'bone.jpg', image)
    except:
        logger.info(name)


def generate_df(base, files):
    train_di = {}

    for filename in tqdm(files):
        path = os.path.join(base, filename)
        dcm = pydicom.dcmread(path)
        all_keywords = dcm.dir()
        ignored = ['Rows', 'Columns', 'PixelData']

        for name in all_keywords:
            if name in ignored:
                continue

            if name not in train_di:
                train_di[name] = []

            train_di[name].append(dcm[name].value)

    df = pd.DataFrame(train_di)

    return df


DATAPATH = '/media/ps/_data/ICH/rsna-master/data/'
# TRAIN_DIR = os.path.join(DATAPATH, 'raw/stage_2_train_images')
# TEST_DIR = os.path.join(DATAPATH, 'raw/stage_2_test_images')
# TRAIN_DIR = os.path.join(DATAPATH, 'dataset/train')
# TEST_DIR = os.path.join(DATAPATH, 'dataset/test')
PATHPROC = os.path.join(DATAPATH, 'proc')
#
# logger.info('Create test meta files')
# test_files = os.listdir(TEST_DIR)
# test_df = generate_df(TEST_DIR, test_files)
# test_df.to_csv(os.path.join(DATAPATH, 'test_metadata.csv'))
#
# logger.info('Create train meta files')
# train_files = os.listdir(TRAIN_DIR)
# train_df = generate_df(TRAIN_DIR, train_files)
# train_df.to_csv(os.path.join(DATAPATH, 'train_metadata.csv'))
#
# logger.info('Load meta files')
trnmdf = pd.read_csv(os.path.join(DATAPATH, 'new_train_metadata.csv'))
logger.info('Train meta shape {} {}'.format(*trnmdf.shape))
#
tstmdf = pd.read_csv(os.path.join(DATAPATH, 'new_test_metadata.csv'))
logger.info('Test  meta shape {} {}'.format(*tstmdf.shape))
#
mdf = pd.concat([trnmdf, tstmdf], 0)
rescaledict = mdf.set_index('SOPInstanceUID')[['RescaleSlope', 'RescaleIntercept']].to_dict()

logger.info('Create windowed images')
# name =
list = ['ID_0a3c836fb.dcm']
for name in list:
    convert_dicom_to_jpg(name)
# with zipfile.ZipFile(os.path.join(DATAPATH, "raw/rsna-intracranial-hemorrhage-detection.zip"), "r") as f:
#     for t, name in enumerate(tqdm(f.namelist())):


