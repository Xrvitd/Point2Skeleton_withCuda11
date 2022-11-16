import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
import DistFunc as DF
import numpy as np
import pytorch3d as p3d
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from scipy import spatial


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()

        self.normal_channel = normal_channel
        self.num_skel_points = 70

        input_channels = 6
        cvx_weights_modules = []

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(16))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(64))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(64))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(16))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Conv1d(in_channels=16, out_channels=num_class, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(num_class))
        cvx_weights_modules.append(nn.Softmax(dim=1))

        self.cvx_segmask_mlp = nn.Sequential(*cvx_weights_modules)


    def forward(self, xyz,num_class):
        B, _, _ = xyz.shape
        #B N=70 6
        #combine dim 1 and 2
        # xyz = xyz.view(B, -1)

        # combine_feature = torch.zeros((B, num_class,self.num_skel_points*6)).cuda()
        # for i in range(num_class):
        #     combine_feature[:,i,:] = xyz
        # combine_feature = combine_feature.transpose(1,2)


        masks = self.cvx_segmask_mlp(xyz.transpose(1,2))  # need transpose?
        # masks = masks.transpose(1,2)



        return masks

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, xyz,num_class,masks):
        B, _, _ = xyz.shape
        norm = xyz[:, :, 3:6]
        xyz = xyz[:, :, :3]
        # B N=70 6   masks B num_class=10 70
        num_point = xyz.shape[1]
        pointsWithLabel = torch.zeros((B, num_point, 7),requires_grad=True).cuda()
        for i in range(B):
            for j in range(num_point):
                pointsWithLabel[i, j, 0:3] = xyz[i, j, 0:3]
                pointsWithLabel[i, j, 3:6] = norm[i, j, 0:3]
                dist, idx = torch.max(masks[i, :, j], 0)
                pointsWithLabel[i, j, 6] = idx
        pointsByLabel = []
        for i in range(B):
            pointsByLabel.append([])
            for j in range(num_class):
                pointsByLabel[i].append(pointsWithLabel[i, pointsWithLabel[i, :, 6] == j, :])
        loss_points = 0
        loss_norm = 0
        for i in range(B):
            for j in range(num_class):
                if pointsByLabel[i][j].shape[0] > 0:
                    for k in range(pointsByLabel[i][j].shape[0]):
                        loss_points += torch.sum(torch.norm(pointsByLabel[i][j][:, 0:3] - pointsByLabel[i][j][k, 0:3],dim = 1,keepdim= True )) / pointsByLabel[i][j].shape[0]
                        loss_norm += torch.sum(torch.norm(pointsByLabel[i][j][:, 3:6] - pointsByLabel[i][j][k, 3:6],dim = 1,keepdim= True )) / pointsByLabel[i][j].shape[0]
        loss_points = loss_points / (B*num_class)
        loss_norm = loss_norm / (B*num_class)





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

