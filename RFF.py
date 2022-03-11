import torch
from torch import nn

class RandomFeatureMap(nn.Module):
    
    def __init__(self,nfeature,d_model):
        """
        Parameters
        ----------
        nfeature : int
            Number of random features.
        d_model : int
            dataset feature dimension.

        """
        super(RandomFeatureMap,self).__init__()
        self.nfeature = nfeature
        self.d_model = d_model
    
    
    def forward(self,x,feature):
        """
        Parameters
        ----------
        x : Tensor, no grad
            
            Dataset of dimension [bsz,ndata,d_model]
            
        feature : Tensor
        
            Random feature tensor of dimension [bsz,d_model,nfeature]. Should be replicas along the 1st dimension, i.e,
            feature[i,:,:] = feature[j,:,:]

        Returns
        -------
        Tensor
            Tensor of dimension [bsz, ndata, 2*nfeature]. The basis tensor that could be used in any base learning task.

        """
        
        x = torch.einsum("bij,bjk -> bik",x,feature)
        x1 = torch.cos(x)/torch.sqrt(self.nfeature)
        x2 = torch.sin(x)/torch.sqrt(self.nfeature)
        
        return torch.cat((x1,x2),2)
        
