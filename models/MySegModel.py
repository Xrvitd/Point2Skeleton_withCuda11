import torch
import torch.nn as nn
import torch.nn.functional as F
import DistFunc as DF
import numpy as np
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from scipy import spatial


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        # self.num_skel_points = 70

        self.sa0 = PointNetSetAbstractionMsg(1024, [0.1, 0.2], [16, 32], in_channel, [[16, 16, 32], [16, 16, 32]])
        self.sa1 = PointNetSetAbstractionMsg(768, [0.2, 0.4], [32, 64], 32 * 2, [[32, 32, 64], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.4, 0.6], [32, 64], 64 * 2, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(512, [0.6, 0.8], [64, 128], 128 * 2, [[128, 128, 256], [128, 128, 256]])

        input_channels = 256 + 256
        cvx_weights_modules = []

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=input_channels, out_channels=384, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(384))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=384, out_channels=256, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(256))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(256))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(128))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Conv1d(in_channels=128, out_channels=num_class, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(num_class))
        cvx_weights_modules.append(nn.Softmax(dim=2))

        self.cvx_weights_mlp = nn.Sequential(*cvx_weights_modules)


    def forward(self, xyz,num_class):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l0_xyz, l0_points = self.sa0(xyz, norm)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        context_features = l3_points
        weights = self.cvx_weights_mlp(context_features)  # need transpose?
        l3_xyz = l3_xyz.transpose(1, 2)
        skel_xyz = torch.sum(weights[:, :, :, None] * l3_xyz[:, None, :, :], dim=2)

        return skel_xyz

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, xyz,num_class,skel_xyz):
        B, _, _ = xyz.shape
        norm = xyz[:, :, 3:6]
        xyz = xyz[:, :, :3]
        # B N=70 6   masks B num_class=10 70
        num_point = xyz.shape[1]
        knn_shape = DF.knn_with_batch(xyz,xyz,15)
        changingrate = torch.zeros((B,num_point),requires_grad=True).cuda()

        for i in range(B):
            for j in range(num_point):
                changingrate[i,j]=torch.sum(torch.min(torch.norm(torch.linalg.cross(norm[i, j, None,:], norm[i, knn_shape[i, j, :], :]),dim = 1),torch.norm(torch.mul(norm[i, j, None,:], norm[i, knn_shape[i, j, :], :]),dim = 1)))

        with open('changingrate.txt' , "w") as f:
            i=0
            for j in range(num_point):
                f.write("%f %f %f %f %f %f\n" % (
                xyz[i, j, 0], xyz[i, j, 1], xyz[i, j, 2],
                (changingrate[i, j] - torch.min(changingrate[i, :])) / (torch.max(changingrate[i, :])- torch.min(changingrate[i, :]))*2,0,0))

        loss_cross =0
        # for i in range(xyz.shape[0]):
        #     for j in range(num_class):
        #         norm_tmp = norm[i,:,:]*masks[i,j,:].view(-1,1)  #?
        #         norm_tmp = norm_tmp*xyz.shape[1]
        #         loss_nor = torch.zeros(int((norm_tmp.shape[0]* (norm_tmp.shape[0]-1))/2),3).cuda()
        #         id =0
        #         for k in range(norm_tmp.shape[0]):
        #             for l in range(k + 1, norm_tmp.shape[0]):
        #                 loss_nor[id, :] = torch.cross(norm_tmp[k,:],norm_tmp[l,:])
        #                 id = id+1
        #         avg_nor = torch.mean(loss_nor,0)
        #         for k in range(loss_nor.shape[0]):
        #             loss_cross += torch.cross(loss_nor[k,:],avg_nor).norm()
        #
        # loss_cross = loss_cross/(xyz.shape[0]*num_class*loss_nor.shape[0])

        loss_chosen = 0
        # for i in range(xyz.shape[0]):
        #     for k in range(xyz.shape[1]):
        #         # maskkk = 0
        #         for j in range(num_class):
        #             # maskkk += masks[i,j,k]
        #             loss_chosen += 1.0-(1.0-masks[i,j,k]*2.0)**2
        # loss_chosen = loss_chosen/(xyz.shape[0]*xyz.shape[1]*num_class)
        #

        # loss combination
        # print('loss_normal', loss_normal1-loss_normal11)
        final_loss = 1.0*loss_points + 1.5*loss_norm
        # final_loss = 1.0*loss_cross + 1.0*loss_chosen


        return final_loss

