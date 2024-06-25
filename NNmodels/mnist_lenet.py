import torch.nn.functional as F
from torch import nn
import numpy as np

# import torch
class lenet(nn.Module):
    def __init__(self,args):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

# class lenet(nn.Module):
#
#     def __init__(self, input_channels, output_channels):
#         super(lenet, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320,50)
#         self.fc2 = nn.Linear(50, output_channels)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.conv2_drop(x)
#         x = F.max_pool2d(x,2)
#         x = F.relu(x)
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return  x


# class mnistlenet(nn.Module): 					# 继承于nn.Module这个父类
#     def __init__(self,args):						# 初始化网络结构
#         super(mnistlenet, self).__init__()    	# 多继承需用到super函数
#         self.block_1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 输出为6*28*28
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为6*14*14
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 输出为16*10*10
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为16*5*5
#         )
#         self.block_2 = nn.Sequential(
#             nn.Linear(16*5*5, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, 10),
#         )
#
#     def forward(self, x):  # 正向传播过程
#         x = self.block_1(x)
#         x = x.view(-1,16*5*5)
#         x = self.block_2(x)
#         return x

class mnistlenet(nn.Module):

    # network structure
    def __init__(self, args):
        super(mnistlenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)