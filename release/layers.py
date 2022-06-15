import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
device = torch.device("cuda:0")
def func_MProduct(C, M):
    assert C.size()[0] == M.size()[0]
    Tr = C.size()[0]
    N = C.size()[1]
    C_new = torch.sparse.FloatTensor(C.size()).cuda()
    #C_new = C.clone()
    for j in range(Tr):
        idx = C._indices()[0] == j
        mat = torch.sparse.FloatTensor(C._indices()[1:3,idx], C._values()[idx], torch.Size([N,N])).cuda()
        tensor_idx = torch.zeros([3, mat._nnz()], dtype=torch.long).cuda()
        tensor_val = torch.zeros([mat._nnz()], dtype=torch.float).cuda()
        tensor_idx[1:3] = mat._indices()[0:2]
        indices = torch.nonzero(M[:,j])
        # assert indices.size()[0] <= no_diag
        for i in range(indices.size()[0]):
            tensor_idx[0] = indices[i]
            tensor_val = M[indices[i], j] * mat._values()
            # print(C_new.device,tensor_idx.device,tensor_val.device)
            temp = torch.sparse.FloatTensor(tensor_idx, tensor_val, C.size()).cuda().detach()
            C_new = C_new.clone() + temp
        C_new.coalesce()                      
    return C_new

def func_MProduct_dense(C, M):
    T = C.shape[0]
    X = torch.matmul(M, C.reshape(T, -1)).reshape(C.size())
    # indices = torch.nonzero(X).t()
    # values = X[indices[0],indices[1],indices[2]] # modify this based on dimensionality
    # Cm = torch.sparse.FloatTensor(indices, values, X.size())
    return X

def fft_product(X,  W, sparse = False):
    if sparse is False:
        # X = X.permute(1,2,0)
        # W = W.permute(1,2,0)
        Xf = torch.fft.fft(X)
        Wf = torch.fft.fft(W)
        temp = torch.fft.ifft(torch.einsum('ijk,jrk->irk',Xf,Wf))
    else :
        # X = X.to_dense()
        # X = X.permute(1,2,0)
        # W = W.permute(1,2,0)
        Xf = torch.fft.fft(X)
        Wf = torch.fft.fft(W)
        temp = torch.fft.ifft(torch.einsum('ijk,jrk->irk',Xf,Wf))
    return temp.real

def compute_AtXt(A, X, M):
    # X = X.transpose(2, 0)
    # W = W.transpose(2, 0)
    At = func_MProduct(A, M)
    Xt = func_MProduct_dense(X, M)
    # print(Xt, At)
    AtXt = torch.zeros(X.shape[0], X.shape[1], X.shape[-1]).cuda()
    AX = AtXt
    for k in range(X.shape[0]):
        AtXt[k] = torch.sparse.mm(At[k], Xt[k])
    
    # AX = func_MProduct_dense(AtXt,M.t())
    return AX

def phi_product2(A, X):
    # X = X.transpose(2, 0)
    # W = W.transpose(2, 0)
    temp = torch.einsum('ijk,jrk->irk',A,X)
    return temp  

class TensorGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, transm, numnodes, out_features, channel_nums, bias=True):
        super(TensorGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_features = channel_nums
        self.numnodes = numnodes
        self.transm = transm
        self.weight = Parameter(torch.FloatTensor(channel_nums,in_features, out_features))
        # self.transform = Parameter(torch.FloatTensor(channel_nums, channel_nums))
        if bias:
            self.bias = Parameter(torch.FloatTensor(channel_nums,numnodes,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #self.transform.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = fft_product(input, self.weight, sparse=False)
        # output = fft_product(adj, support, sparse=True)
        support = compute_AtXt(adj, input, self.transm)
        # print(support.size(),self.weight.size())
        output = torch.matmul(support, self.weight)
        # print(output.size(), self.bias.size())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TensorGraphConvolutionFFT(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, transm, numnodes, out_features, channel_nums, bias=True):
        super(TensorGraphConvolutionFFT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_features = channel_nums
        self.numnodes = numnodes
        self.transm = transm
        self.weight = Parameter(torch.FloatTensor(in_features, out_features,channel_nums))
        # self.transform = Parameter(torch.FloatTensor(channel_nums, channel_nums))
        if bias:
            self.bias = Parameter(torch.FloatTensor(numnodes,out_features,channel_nums))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #self.transform.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = fft_product(input, self.weight, sparse=False)
        output = fft_product(adj, support, sparse=True)
        # support = compute_AtXt(adj, input, self.transm)
        # print(support.size(),self.weight.size())
        # output = torch.matmul(support, self.weight)
        # print(output.size(), self.bias.size())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class TensorLinear(Module):
    def __init__(self, in_features, numnodes, out_features, nclass, bias=True):    
        super(TensorLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.numnodes = numnodes
        self.nclass = nclass
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(numnodes,nclass))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data = torch.rand(self.weight.size())
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # self.weight.data[self.weight.data < 0]=0
        # self.weight.data[self.weight.data > 1]=1
        # self.weight.data = self.weight.data/self.weight.data.sum()
        output = torch.einsum('ijk,kr->ijr',input,self.weight)
        output = torch.reshape(output, (output.shape[0],output.shape[1]))
        # print(self.weight.data)
        if self.bias is not None:
            return output + self.bias
        else:
            return output