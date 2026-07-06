import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MentalHealthResNet(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        # 使用 ResNet18 作为 Backbone
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 替换最后的全连接层
        n_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_inputs, num_classes)

    def forward(self, x):
        return self.resnet(x)