import torch

device = 'cuda'
seed = 1234
fold = 1
epochs = 20
start = 0
test_batchsize = 128
batchsize = 42  # 32
lr = 0.00005  # 0.00001
size = 256
hflip = True  # Augmentation - Embedding horizontal flip
transpose = True
autocrop = True
loss = 'bce'
model = 'resnet50'
dataset = '/media/ps/_data1/ICH/ich-master/dataset/'
emb_pth = "/media/ps/_data1/ICH/ich-master/experiment/10-5-test/emb/"  # emb保存地址
model_save_path = "/media/ps/_data1/ICH/ich-master/experiment/10-5-test/"  # 模型权重保存路径
load_path = "/media/ps/_data1/ICH/ich-master/experiment/r50+d121+bce+1.5*(l1+l2)/weights/modelresnet50_256_epoch10.bin"
infer = 'EMB'
label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']  # , 'healthy'
n_classes = 6
val = False
n_gpu =torch.cuda.device_count()
load_model = False
iswandb = True  #
freeze = False
weight_dacay = 0.0001
lr_gamma = 0.5
lr_epoch = 3
model1 = 'resnet50'
model2 = 'densenet121'
load_start = 13
train_strategy = 'all'  # single or all
#
if __name__ == '__main__':
#     print(torch.cuda.device_count())
   print(n_gpu)
