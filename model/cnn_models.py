import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest101

from configs import config_cnn
from model import resnet, repvgg, pvt, rnn_models
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from model import compact_bilinear_pooling
from model import cbp
import os
from thop import profile
from ptflops import get_model_complexity_info


n_classes = config_cnn.n_classes


class Mynet(nn.Module):
    def __init__(self, backbone1, backbone2, freeze=False):
        super(Mynet, self).__init__()
        self.backbone1 = nn.Sequential(*list(backbone1.children())[:-2])
        self.backbone2 = nn.Sequential(*list(backbone2.children())[:-1])
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out2 = nn.Linear(2048, 6)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out3 = nn.Linear(1024, 6)
        self.cbp_layer = cbp.CompactBilinearPooling(2048, 1024, 8192)
        self.fc1 = torch.nn.Linear(8192, 2048)  # 8192
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024, 6)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc1.weight.data)
        if self.fc1.bias is not None:
            torch.nn.init.constant_(self.fc1.bias.data, val=0)
        torch.nn.init.kaiming_normal_(self.fc2.weight.data)
        if self.fc2.bias is not None:
            torch.nn.init.constant_(self.fc2.bias.data, val=0)
        torch.nn.init.kaiming_normal_(self.fc3.weight.data)
        if self.fc3.bias is not None:
            torch.nn.init.constant_(self.fc3.bias.data, val=0)
        # 两层全连接层
        # torch.nn.init.kaiming_normal_(self.fc2.weight.data)
        # if self.fc2.bias is not None:
        #     torch.nn.init.constant_(self.fc2.bias.data, val=0)

    def forward(self, x):
        ##同一个loss的情况
        y1 = self.backbone1(x)
        y2 = self.backbone2(x)
        output1 = self.cbp_layer(y1, y2)
        output1 = F.normalize(output1, dim=1)
        output1 = output1.squeeze()
        output1 = self.fc1(output1)
        output1 = self.fc2(output1)
        output1 = self.fc3(output1)
        # return output1

        ##每一个网络有不同的loss
        output2 = self.avgpool2(y1)
        output2 = output2.view(x.size(0), -1)
        output2 = self.fc_out2(output2)

        output3 = self.avgpool3(y2)
        output3 = output3.view(x.size(0), -1)
        output3 = self.fc_out3(output3)

        return output1, output2, output3

    def features(self, x):
        y1 = self.backbone1(x)
        y2 = self.backbone2(x)
        output1 = self.cbp_layer(y1, y2)
        output1 = F.normalize(output1, dim=1)
        # output1 = output1.squeeze()
        return output1


class Tongyuan_cbp(nn.Module):
    def __init__(self, backbone, freeze=False):
        super(Tongyuan_cbp, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # self.backbone2 = nn.Sequential(*list(backbone.children())[:-2])
        # self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        # self.fc_out2 = nn.Linear(2048, 6)
        # self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        # self.fc_out3 = nn.Linear(2048, 6)
        self.cbp_layer = cbp.CompactBilinearPooling(2048, 2048, 4196)
        self.fc1 = torch.nn.Linear(4196, 2048)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024, 6)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc1.weight.data)
        if self.fc1.bias is not None:
            torch.nn.init.constant_(self.fc1.bias.data, val=0)
        torch.nn.init.kaiming_normal_(self.fc2.weight.data)
        if self.fc2.bias is not None:
            torch.nn.init.constant_(self.fc2.bias.data, val=0)
        torch.nn.init.kaiming_normal_(self.fc3.weight.data)
        if self.fc3.bias is not None:
            torch.nn.init.constant_(self.fc3.bias.data, val=0)
        # 两层全连接层
        # torch.nn.init.kaiming_normal_(self.fc2.weight.data)
        # if self.fc2.bias is not None:
        #     torch.nn.init.constant_(self.fc2.bias.data, val=0)

    def forward(self, x):
        ##同一个loss的情况
        y1 = self.backbone(x)
        y2 = self.backbone(x)
        output = self.cbp_layer(y1, y2)
        output = F.normalize(output, dim=1)
        output = output.squeeze()
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
        ##每一个网络有不同的loss
        # output2 = self.avgpool2(y1)
        # output2 = output2.view(x.size(0), -1)
        # output2 = self.fc_out2(output2)
        # output3 = self.avgpool3(y2)
        # output3 = output3.view(x.size(0), -1)
        # output3 = self.fc_out3(output3)
        # return output1,output2,output3

    def features(self, x):
        y1 = self.backbone1(x)
        y2 = self.backbone2(x)
        output1 = self.cbp_layer(y1, y2)
        output1 = F.normalize(output1, dim=1)
        output1 = output1.squeeze()
        return output1


class NoTongyuan_cbp(nn.Module):
    def __init__(self, backbone1, backbone2, freeze=False):
        super(NoTongyuan_cbp, self).__init__()
        self.backbone1 = nn.Sequential(*list(backbone1.children())[:-2])
        self.backbone2 = nn.Sequential(*list(backbone2.children())[:-2])
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out2 = nn.Linear(2048, 6)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out3 = nn.Linear(2048, 6)
        self.cbp_layer = cbp.CompactBilinearPooling(2048, 2048, 8192)
        self.fc1 = torch.nn.Linear(8192, 2048)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024, 6)

    def forward(self, x):
        ##同一个loss的情况
        y1 = self.backbone1(x)
        y2 = self.backbone2(x)
        output = self.cbp_layer(y1, y2)
        output = F.normalize(output, dim=1)
        output = output.squeeze()
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
        ##每一个网络有不同的loss

    def features(self, x):
        y1 = self.backbone1(x)
        y2 = self.backbone2(x)
        output1 = self.cbp_layer(y1, y2)
        output1 = F.normalize(output1, dim=1)
        output1 = output1.squeeze()
        return output1


class Mynet_zijiwan(nn.Module):
    def __init__(self, backbone, freeze=False):
        super(Mynet_zijiwan, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-3])
        self.layer1 = nn.Sequential(*list(backbone.children())[-3:-2])
        self.layer2 = nn.Sequential(*list(backbone.children())[-3:-2])
        # self.backbone2 = nn.Sequential(*list(backbone1.children())[:-2])
        self.cbp_layer = cbp.CompactBilinearPooling(2048, 1024, 8192)
        self.fc = torch.nn.Linear(8192, 6)
        # self.fc2 = torch.nn.Linear(2048, 6)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
        # 两层全连接层
        # torch.nn.init.kaiming_normal_(self.fc2.weight.data)
        # if self.fc2.bias is not None:
        #     torch.nn.init.constant_(self.fc2.bias.data, val=0)

    def forward(self, x):
        # compact_bilinear_pooling_部分共享
        x = self.backbone(x)
        y1 = self.layer1(x)
        y2 = self.layer2(x)
        # batch_size,emb, height, width = x.size()
        output = self.cbp_layer(y1, y2)
        output = F.normalize(output, dim=1)
        output = output.squeeze()
        output = self.fc(output)
        return output


class zhengliu(nn.Module):
    def __init__(self, backbone1, backbone2):
        super(zhengliu, self).__init__()
        self.cnn1 = backbone1  # nn.Sequential(*list(backbone1.children()))
        self.cnn1.fc = nn.Linear(2048, 6)
        self.cnn2 = backbone2  # nn.Sequential(*list(backbone2.children()))
        self.cnn2.classifier = nn.Linear(1024, 6)

    def forward(self, x):
        output1 = self.cnn1(x)
        output2 = self.cnn2(x)
        # output1 = self.cnn1.ext_features(x)
        # output2 = self.cnn2.features(x)
        return output1, output2
        # self.feature1 = self.cnn1.ext_features()
        # self.feature2

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    torch.cuda.set_device(0)
    imgs = torch.rand(1, 3, 256, 256).cuda()

    # cnn1 = resnet.resnet50(pretrained=True).cuda()
    # cnn2 = torchvision.models.densenet121(pretrained=True).cuda()
    # model = Mynet(cnn1, cnn2).cuda()

    # model = EfficientNet.from_pretrained('efficientnet-b6').cuda()

    # model = resnest101(pretrained=True)

    # model = repvgg.create_RepVGG_B1(deploy=False)

    # model = pvt.pvt_medium(pretrained=True)

    model = rnn_models.NeuralNet(embed_size=8192, LSTM_UNITS=2048, DO=0.2)

    macs, params = get_model_complexity_info(model, (14,8192), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(params)
    print(macs)


