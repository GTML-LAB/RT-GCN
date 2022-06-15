from __future__ import division
from __future__ import print_function
from re import sub

import time
import argparse
import numpy as np
from torch_geometric.utils import dropout_adj
from deeprobust.graph.data import Dataset, PrePtbDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from utils import *
from models import TenGCN, TenGCNFFT
from TRPCA_torch import TRPCA
from copy import deepcopy
import pickle
import warnings
import numpy as np
import sklearn
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import seaborn as sb
output_features = None
warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora_ml',
                    help='Dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass')
# parser.add_argument('--seed', type=int, default=115, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability)')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])                    
parser.add_argument('--ptb_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--svd_rank', type=int,  default=100, help='rank for svd decomposition')
parser.add_argument('--ratio', type=float,  default=0.9, help='rank for svd decomposition')
parser.add_argument('--num_subadj', type=int,  default=3, help='number of adj2 layers')
parser.add_argument('--tsne_confuse', type=int,  default=20, help='number of tsne_confuse, a parameter of tsne function')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
dataset = args.dataset
epochs = args.epochs
ratio = args.ratio
acc_results = []

times = 10

def plot(x, colors, path=None):
    palette = np.array(sb.color_palette("hls", 10))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.savefig(path)
    plt.show()

def load_data():
    data = Dataset(root='../Data/', name=args.dataset, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    labels = torch.LongTensor(labels)
    return data, adj, features, labels, idx_train, idx_val, idx_test


for k in range(times):
    # random seed
    np.random.seed()
    # Load data
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data()

    ## attacking data start
    # w/o attack
    if args.attack == 'no':
        perturbed_adj = adj
    # random attack
    if args.attack == 'random':
        from deeprobust.graph.global_attack import Random
        attacker = Random()
        n_perturbations = int(args.ptb_rate * (adj.sum()//2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj
    # meta or nettack attack
    if args.attack == 'meta' or args.attack == 'nettack':
        perturbed_data = PrePtbDataset(root='../Data/',
                name=args.dataset,
                attack_method=args.attack,
                ptb_rate=args.ptb_rate if args.ptb_rate > 0 else 1.0)
        perturbed_adj = perturbed_data.adj if args.ptb_rate > 0 else adj
        if args.attack == 'nettack':
            idx_test = perturbed_data.get_target_nodes()

    if args.dataset == 'polblogs':
        import scipy
        features = perturbed_adj + scipy.sparse.csr_matrix(np.eye(perturbed_adj.shape[0]))
    ## attacking data end

    # processing f and adj
    features = normalize(features)
    # perturbed_adj = normalize_adj(perturbed_adj)
    features = torch.FloatTensor(np.array(features.todense()))
    # perturbed_adj = adj

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # print(type(features))
    print("Loaded Done!")

    def adj_sample(edges, ratio):
        edges2 = sparse_mx_to_torch_sparse_tensor(edges)
        indices = edges2._indices()
        indices2 = dropout_adj(indices, p = ratio, force_undirected = True)
        indices2 = indices2[0]
        values = torch.ones(len(indices2[0]))
        shape = torch.Size(edges.shape)
        return torch.sparse.FloatTensor(indices2, values, shape)
    
    def both_aug(edges, ratio):
        edges2 = sparse_mx_to_torch_sparse_tensor(edges)
        indices = edges2._indices()
        indices2 = dropout_adj(indices, p = ratio, force_undirected = True)
        indices2 = indices2[0]
        num = round(len(indices[0])*ratio/2)
        new_index = torch.randint(edges.shape[0],(2,num))
        new_index2 = new_index
        new_index2[0], new_index2[1] = new_index[1], new_index[0]
        indices3 = torch.cat([indices2,new_index, new_index2], 1)
        values = torch.ones(len(indices3[0]))
        shape = torch.Size(edges.shape)
        return torch.sparse.FloatTensor(indices3, values, shape)

    subadj = []
    fea = []
    subadj.append(normalize_adj_tensor(sparse_mx_to_torch_sparse_tensor(perturbed_adj).to_dense()))
    fea.append(features)
    for i in range(args.num_subadj-1):
        subadj.append(normalize_adj_tensor(both_aug(perturbed_adj,1-ratio).to_dense()))
        # subadj.append(both_aug(perturbed_adj,1-ratio).to_dense())
        fea.append(features)
    adj2 = torch.stack(subadj,dim=2)
    features2 = torch.stack(fea, dim=2)

    # T-SVD
    model0 = TRPCA()
    adj2=adj2.cuda()
    L = model0.T_SVD(adj2,args.svd_rank)

    transform_matrix = torch.eye(perturbed_adj.shape[0]).cuda()

    for i in range(L.shape[2]):
        L[:,:,i] = normalize_adj_tensor(L[:,:,i])

    # Model and optimizer
    best_loss_test = 100
    best_acc_test = 0
    model = TenGCNFFT(nfeat=features2.shape[1],
                transm = transform_matrix,
                numnodes = features2.shape[0],
                nhid=args.hidden,
                nchannel = features2.shape[2],
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)


    if args.cuda:
        model.cuda()
        features2 = features2.cuda()
        L = L.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(epoch):
        global best_loss_test
        global best_acc_test
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features2, L)
        
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features2, L)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # + torch.norm(model.gc1.weight, p='fro') 
        acc_val = accuracy(output[idx_val], labels[idx_val])

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        if best_loss_test > loss_test:
            best_loss_test = loss_test
        
        if best_acc_test < acc_test:
            best_acc_test = acc_test

    def test():
        global best_loss_test
        global best_acc_test

        acc_results.append(best_acc_test.detach().cpu().numpy())
        print("Test set results:",
            "loss= {:.4f}".format(best_loss_test),
            "accuracy= {:.4f}".format(best_acc_test))


    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()
    output_features = model.output_features

digits_final = TSNE(perplexity=args.tsne_confuse).fit_transform(output_features.cpu().detach().numpy())
plot(digits_final,
        labels.cpu().detach().numpy(), 
        path="./" + str(args.attack)+"_"+str(args.dataset)+"_"+str(args.ptb_rate)+"_tsne_confuse="+str(args.tsne_confuse)+".png")

print("mean = {:.4f}".format(np.max(acc_results)),"std = {:.4f}".format(np.std(acc_results)))
dataset = args.dataset
attack_mode = args.attack
svd_rank = args.svd_rank
ratio = args.ratio
ptb_rate = args.ptb_rate
num_adj = args.num_subadj
with open("./rst_record.txt", 'a') as f:
        f.write(str(dataset) + "_" + str(attack_mode) + "_" + str(ptb_rate) + "_" + str(svd_rank) + "_" + str(ratio) + "_num_adj:" + str(num_adj) + ": \tmean:" + str(np.max(acc_results)) + "\tvar:" + str(np.std(acc_results)) + "\n")
