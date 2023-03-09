# from hdmapnet 
# /share/wsq/HDMapNet/model
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1) # 上采样
        x1 = torch.cat([x2, x1], dim=1) # 将x2和x1进行拼接，送入网络
        return self.conv(x1)


class BevEncode(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, pretrained=False):
        # 删掉了原来代码中embedding 和 direction的部分
        super(BevEncode, self).__init__()
        trunk = resnet18(weights='ResNet18_Weights.DEFAULT', zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )
        
        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4) 
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )


    def forward(self, x):
        # 输入x的维度 [1,1,1000,1000]
        x = self.conv1(x) # [1,64,500,500]
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # [1,64,500,500]
        x = self.layer2(x1) # [1,128,250,250]
        x2 = self.layer3(x) # [1,256,125,125]

        x = self.up1(x2, x1) # [1,256,500,500]
        x = self.up2(x) # [1,2,1000,1000]
        
        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None


        return x, x_embedded
