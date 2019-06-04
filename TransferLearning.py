import copy
import torch.nn as nn
from collections import OrderedDict

class Resnet_fc(nn.Module):
    def __init__(self, base_model, nb_classes, toFreeze=False):
        super(Resnet_fc, self).__init__()
        
        base_model_copy = copy.deepcopy(base_model)
        self.feature_extractor = nn.Sequential(*list(base_model_copy.children())[:-2])
        
        # toFreeze가 true면 파라미터 업데이트를 안한다(학습X)
        if toFreeze:
          for param in self.feature_extractor.parameters():
            param.requires_grad=False
        else:
          for param in self.feature_extractor.parameters():
            param.requires_grad=True
          
        
        self.gap = nn.AvgPool2d(7, 1)
        self.linear = nn.Linear(2048, nb_classes)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.linear(x)
        
        return x