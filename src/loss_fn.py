from configs import LAMBDA_COORD, LAMBDA_NOOBJ, S, B
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat, y, y_hat):
        pass

    
