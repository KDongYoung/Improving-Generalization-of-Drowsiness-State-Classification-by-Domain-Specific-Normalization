import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.CrossEntropyLoss):
    def __init__(self, **args):
        super(CELoss, self).__init__()
        self.lossfn = torch.nn.CrossEntropyLoss()  
    
    def forward(self, output, target):
        
        loss = self.lossfn(output, target) 
        return loss