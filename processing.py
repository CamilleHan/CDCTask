import copy
import torch
from torch import nn
import tensorflow as tf
import numpy  as np
def process_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    g = np.zeros((len(w_avg.keys()),len(w)))
    a = 0
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], (s_num[i]/(temp_sample_num))).long()
    for k in w_avg.keys():  # the nn layer loop
        for j in range(len(w)):
            # d = np.array(w[j][k].cpu())
            g[a,j] = np.array(torch.norm((w[j][k].float()), p=2).cpu()).reshape(1)
            # gl = np.linalg.norm(d)
            # gl[a] = np.linalg.norm(g[a, j])
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/(temp_sample_num))
        a =+ 1
    glob = np.linalg.norm(g)
    return w_avg,glob
