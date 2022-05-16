import torch
import torch.nn as nn

class RandomFeatureMap(nn.module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,feature):

        x = torch.einsum("dij,djk -> dik",x,feature)
        x_cos = torch.cos(x)
        x_sin = torch.sin(x)
        return torch.cat((x_cos,x_sin),2)