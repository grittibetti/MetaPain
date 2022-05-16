from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class kernel_encoder(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.Linear(2*d, 256)
        self.relu1 = nn.ReLU()
        self.ln2 = nn.Linear(256,256)
        self.relu2 = nn.ReLU()
        self.ln3 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        data_in = torch.cat((x,y),1)
        out = self.ln1(data_in)
        out = self.relu1(out)
        out = self.ln2(out)
        out = self.relu2(out)
        out = self.ln3(out)
        out = self.sigmoid(out)

        return out
