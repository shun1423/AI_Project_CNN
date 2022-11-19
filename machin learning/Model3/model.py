import torch
import torch.nn as nn
from utils import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

## model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4,4))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*4*4, out_features=4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=10, bias=True)
        )

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

## Base Model
class Base_model(nn.Module):
    def __init__(self):
        super(Base_model, self).__init__()
        self.backbone = AlexNet()
        self.backbone.load_state_dict(torch.load('./checkpoint_Q2.pth'))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.classifier[8] = nn.Linear(in_features=1024, out_features=100, bias=True)
        self.backbone.classifier[8].requires_grad = True
        nn.init.kaiming_normal_(self.backbone.classifier[8].weight)

    def forward(self, x):
        out = self.backbone(x)
        return out

## Model 1 - remove conv5, fc2, retrain conv4, fc1,fc3
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.model = AlexNet()
        self.model.load_state_dict(torch.load('./checkpoint_Q2.pth'))

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[8] = nn.Linear(in_features=4096, out_features=100, bias=True)
        self.model.classifier[8].requires_grad = True

        for i in range(14, 17):
            self.model.features[i] = Identity()
        for i in range(5, 8):
            self.model.classifier[i] = Identity()

        for i in [1,2,8]:
            self.model.classifier[i].requires_grad = True
        for i in [11,12]:
            self.model.features[i].requires_grad = True

        nn.init.kaiming_normal_(self.model.classifier[1].weight)
        nn.init.kaiming_normal_(self.model.classifier[2].weight)
        nn.init.kaiming_normal_(self.model.classifier[8].weight)
        nn.init.kaiming_normal_(self.model.features[11].weight)  # conv4
        nn.init.constant_(self.model.features[12].weight, 1)  # BN
        nn.init.constant_(self.model.features[12].bias, 0)  # BN
        # self.model.classifier[8] = nn.Linear(in_features=1024, out_features=100)
        #
        # for i in [1,2,5,6,8]:
        #     self.model.classifier[i].requires_grad = True
        # for i in [11,12,14,15]:
        #     self.model.features[i].requires_grad = True
        #
        # for i in [1, 5, 8]:
        #     nn.init.kaiming_normal_(self.model.classifier[i].weight)
        # for i in [2, 6]:
        #     nn.init.constant_(self.model.classifier[i].weight, 1)
        #     nn.init.constant_(self.model.classifier[i].bias, 0)

    def forward(self, x):
        out = self.model(x)
        return out

## Model 2 - remove conv5, fc2
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.model = AlexNet()
        self.model.load_state_dict(torch.load('./checkpoint_Q2.pth'))

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[8] = nn.Linear(in_features=4096, out_features=100, bias=True)
        self.model.classifier[8].requires_grad = True
        nn.init.kaiming_normal_(self.model.classifier[8].weight)

        for i in range(14, 17):
            self.model.features[i] = Identity()
        for i in range(5, 8):
            self.model.classifier[i] = Identity()

    def forward(self, x):
        out = self.model(x)
        return out

## Model 3 - retrain conv4, conv5, all fc
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.model = AlexNet()
        self.model.load_state_dict(torch.load('./checkpoint_Q2.pth'))

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[8] = nn.Linear(in_features=1024, out_features=100, bias=True)
        for i in [1,2,5,6,8]:
            self.model.classifier[i].requires_grad = True
        for i in [11,12,14,15]:
            self.model.features[i].requires_grad = True

        nn.init.kaiming_normal_(self.model.features[11].weight) # conv4
        nn.init.constant_(self.model.features[12].weight, 1)    # BN
        nn.init.constant_(self.model.features[12].bias, 0)      # BN
        nn.init.kaiming_normal_(self.model.features[14].weight) # conv5
        nn.init.constant_(self.model.features[15].weight, 1)    # BN
        nn.init.constant_(self.model.features[15].bias, 0)      # BN

        for i in [1,5,8]:
            nn.init.kaiming_normal_(self.model.classifier[i].weight)
        for i in [2,6]:
            nn.init.constant_(self.model.classifier[i].weight, 1)
            nn.init.constant_(self.model.classifier[i].bias, 0)

    def forward(self, x):
        out = self.model(x)
        return out



