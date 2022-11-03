import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np

from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False) #
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512) #全连接层
        self.bn1 = nn.BatchNorm1d(512) #批归一化
        self.drop1 = nn.Dropout(0.4) #随机失活 防止过拟合
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()


    def forward(self, pred, target, trans_feat):
        #total_loss = F.nll_loss(pred, target) #损失函数
        # F.mse_loss()
        cc = target
        total_loss=0
        for branch_idx in range(1,2):
            for i in range(20):
                [x,y,z] = pred[branch_idx][i*3:i*3+3]
                [x1, y1, z1] = target[0][i]
                total_loss += F.mse_loss(x, x1)
                total_loss += F.mse_loss(y, y1)
                total_loss += F.mse_loss(z, z1)
                # total_loss += F.l1_loss(y, torch.Tensor([y1]).cuda())
                # total_loss += F.l1_loss(z, torch.Tensor([z1]).cuda())





        # for branch_idx in range(2):
        #     for i in range(20):
        #         [x,y,z] = pred[branch_idx][i*3:i*3+3]
        #         mindis = 100000
        #         for j in range(mymodel.shape[0]):
        #             [x1,y1,z1] = mymodel[j]
        #             dis = (x-x1)**2 + (y-y1)**2 + (z-z1)**2
        #             if dis < mindis:
        #                 mindis = dis
        #         total_loss += mindis
        # for branch_idx in range(2):
        #     for i in range(20):
        #         [x,y,z] = pred[branch_idx][i*3:i*3+3]
        #         mindis = 100000
        #         for j in range(20):
        #             if i != j:
        #                 [x1,y1,z1] = pred[branch_idx][j*3:j*3+3]
        #                 dis = (x-x1)**2 + (y-y1)**2 + (z-z1)**2
        #                 if dis < mindis:
        #                     mindis = dis
        #         total_loss += 1.1*mindis

        # print('Total Loss is:  ',total_loss)


        return total_loss
