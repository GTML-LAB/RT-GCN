from numpy.ma.core import concatenate
import torch.nn as nn
import torch.nn.functional as F
from layers import TensorGraphConvolution, TensorGraphConvolutionFFT,TensorLinear
import torch


class TenGCN(nn.Module):
    def __init__(self, nfeat, transm, numnodes, nhid, nchannel, nclass, dropout):
        super(TenGCN, self).__init__()

        self.gc1 = TensorGraphConvolution(nfeat, transm, numnodes, nhid, nchannel)
        self.gc2 = TensorGraphConvolution(nhid, transm, numnodes, nclass, nchannel)
        # self.gc3 = nn.Linear(nhid, numnodes,nclass, nchannel)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class TenGCNFFT(nn.Module):
    def __init__(self, nfeat, transm, numnodes, nhid, nchannel, nclass, dropout):
        super(TenGCNFFT, self).__init__()

        self.gc1 = TensorGraphConvolutionFFT(nfeat, transm, numnodes, nhid, nchannel)
        self.gc2 = TensorGraphConvolutionFFT(nhid, transm, numnodes, nclass, nchannel)
        self.nn3 = TensorLinear(nchannel, numnodes, 1,nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj) #2708*7*4
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2]))
        x = self.nn3(x)
        self.output_features = x
        return F.log_softmax(x, dim=1)