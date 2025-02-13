import torch
import torch.nn as nn
from torchgeo.models import FarSeg
from utils import DepthwiseSeparableConv

class FarSegNetwork(nn.Module):
    def __init__(self, num_classes=33, backbone="resnet18", pretrained=True, mobilenets=False):
        super(FarSegNetwork, self).__init__()
        
        self.model = FarSeg(backbone=backbone, classes=num_classes, backbone_pretrained=pretrained)
        
        self.model.backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False) # to adapt to the 12 spectral bands in the dataset
        self.model.decoder.classifier[1] = nn.AdaptiveAvgPool2d((120, 120)) # to match the size of the ground truth masks

        # lightweight Mobilenets architecture
        if mobilenets:
            # Alredy done in the prevoius line
            self.model.backbone.conv1 = DepthwiseSeparableConv(12, 64, kernel_size=7, stride=2, padding=3)

            #First Layer
            self.model.backbone.layer1[0].conv1 = DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer1[0].conv2 = DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer1[1].conv1 = DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer1[1].conv2 = DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1, padding=1)

            #Second Layer
            self.model.backbone.layer2[0].conv1 = DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1)
            self.model.backbone.layer2[0].conv2 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer2[1].conv1 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer2[1].conv2 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1)

            #Third Layer
            self.model.backbone.layer3[0].conv1 = DepthwiseSeparableConv(128, 256, kernel_size=3, stride=2, padding=1)
            self.model.backbone.layer3[0].conv2 = DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer3[1].conv1 = DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer3[1].conv2 = DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1)

            #Fourth Layer
            self.model.backbone.layer4[0].conv1 = DepthwiseSeparableConv(256, 512, kernel_size=3, stride=2, padding=1)
            self.model.backbone.layer4[0].conv2 = DepthwiseSeparableConv(512, 512, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer4[1].conv1 = DepthwiseSeparableConv(512, 512, kernel_size=3, stride=1, padding=1)
            self.model.backbone.layer4[1].conv2 = DepthwiseSeparableConv(512, 512, kernel_size=3, stride=1, padding=1)

            #FPN
            for i in range(4):
                self.model.fpn.layer_blocks[i][0] = DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1)

            #Decoder
            self.model.decoder.blocks[0][0][0] = DepthwiseSeparableConv(256, 128, kernel_size=3, stride=1, padding=1)
            self.model.decoder.blocks[1][0][0] = DepthwiseSeparableConv(256, 128, kernel_size=3, stride=1, padding=1)
            self.model.decoder.blocks[2][0][0] = DepthwiseSeparableConv(256, 128, kernel_size=3, stride=1, padding=1)
            self.model.decoder.blocks[3][0][0] = DepthwiseSeparableConv(256, 128, kernel_size=3, stride=1, padding=1)

            #Classifier
            self.model.decoder.classifier[0] = DepthwiseSeparableConv(128, 33, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.model(x)
