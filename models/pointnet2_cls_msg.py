import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
import DistFunc as DF

from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction



class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.num_skel_points = 100

        self.sa0 = PointNetSetAbstractionMsg(1024, [0.1, 0.2], [16, 32], in_channel, [[16, 16, 32],[16, 16, 32]])
        self.sa1 = PointNetSetAbstractionMsg(768, [0.2, 0.4], [32, 64], 32*2, [[32, 32, 64],[32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.4, 0.6], [32, 64], 64*2, [[64, 64, 128],[64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(512, [0.6, 0.8], [64, 128], 128*2, [[128, 128, 256],[128, 128, 256]])
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
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

        cvx_weights_modules.append(nn.Conv1d(in_channels=128, out_channels=self.num_skel_points, kernel_size=1))
        cvx_weights_modules.append(nn.BatchNorm1d(self.num_skel_points))
        cvx_weights_modules.append(nn.Softmax(dim=2))

        self.cvx_weights_mlp = nn.Sequential(*cvx_weights_modules)

        # self.fc1 = nn.Linear(1024, 512)
        # self.fc1_ = nn.Linear(512, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.1)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc2_ = nn.Linear(256, 256)
        # self.fc2__ = nn.Linear(256, 256)
        # self.fc2___ = nn.Linear(256, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn2_ = nn.BatchNorm1d(256)
        # self.bn2__ = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.1)
        # self.drop3 = nn.Dropout(0.1)
        # self.drop4 = nn.Dropout(0.1)
        # self.drop5 = nn.Dropout(0.1)
        # self.drop6 = nn.Dropout(0.1)
        #
        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
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
        weights = self.cvx_weights_mlp(context_features) #need transpose?
        l3_xyz = l3_xyz.transpose(1,2)
        #skeletal points
        skel_xyz = torch.sum(weights[:, :, :, None] * l3_xyz[:, None, :, :], dim=2)

        # radii
        min_dists, min_indices = DF.closest_distance_with_batch(l3_xyz, skel_xyz, is_sum=False)
        skel_r = torch.sum(weights[:, :, :, None] * min_dists[:, None, :, None], dim=2)

        # surface features
        shape_cmb_features = torch.sum(weights[:, None, :, :] * context_features[:, :, None, :], dim=3)
        shape_cmb_features = shape_cmb_features.transpose(1, 2)

        # x = l3_points.view(B, 1024)

        # x = ((self.fc1(x)))
        # x = ((self.fc2(x)))
        # x = self.fc3(x)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.drop3(F.relu(self.bn2_(self.fc2_(x))))
        # x = self.drop4(F.relu(self.bn2__(self.fc2__(x))))
        # x = self.fc3(x)

        # x = self.fc0(xyz.reshape(1, -1))
        # x = (((self.fc1(x))))
        # x = ((self.fc1_(x)))
        # x = (((self.fc2(x))))
        # x = (((self.fc2_(x))))
        # x = (((self.fc2__(x))))
        # # x = (((self.fc2___(x))))
        # x = self.fc3(x)
        # x = self.fc0(xyz.view(3076))
        # x = self.bn1(self.fc1(x))
        # x = self.bn2(self.fc2(x))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        # return x,l3_points
        return skel_xyz, skel_r, shape_cmb_features


class get_loss_pre(nn.Module):
    def __init__(self):
        super(get_loss_pre, self).__init__()

    def forward(self, shape_xyz, skel_xyz):

        cd1 = DF.closest_distance_with_batch(shape_xyz, skel_xyz)
        cd2 = DF.closest_distance_with_batch(skel_xyz, shape_xyz)
        loss_cd = cd1 + cd2
        loss_cd = loss_cd * 0.0001

        return loss_cd





class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def compute_loss_pre(self, shape_xyz, skel_xyz):

        cd1 = DF.closest_distance_with_batch(shape_xyz, skel_xyz)
        cd2 = DF.closest_distance_with_batch(skel_xyz, shape_xyz)
        loss_cd = cd1 + cd2
        loss_cd = loss_cd * 0.0001

        return loss_cd
    def Rotate_Point3d(self,point,aix,angle):
        # point: [x,y,z]
        # aix: [x,y,z]
        # angle: radian
        # return: [x,y,z]
        aix = aix/aix.norm()
        x = point[0]
        y = point[1]
        z = point[2]
        u = aix[0]
        v = aix[1]
        w = aix[2]
        cos = torch.cos(angle) #弧度制
        sin = torch.sin(angle)
        R = torch.tensor([[u*u+(1-u*u)*cos,u*v*(1-cos)-w*sin,u*w*(1-cos)+v*sin],
                          [u*v*(1-cos)+w*sin,v*v+(1-v*v)*cos,v*w*(1-cos)-u*sin],
                          [u*w*(1-cos)-v*sin,v*w*(1-cos)+u*sin,w*w+(1-w*w)*cos]],requires_grad = True).cuda()
        point = torch.tensor([x,y,z],requires_grad = True).cuda()
        point = torch.matmul(R,point)
        return point

    def forward(self, skel_xyz, skel_radius, shape_cmb_features, shape_xyz,A, w1, w2, w3=0, lap_reg=False):
        #point2skeleton loss
        bn = skel_xyz.size()[0]
        shape_pnum = float(shape_xyz.size()[1])
        skel_pnum = float(skel_xyz.size()[1])

        # sampling loss
        e = 0.57735027
        sample_directions = torch.tensor(
            [[e, e, e], [e, e, -e], [e, -e, e], [e, -e, -e], [-e, e, e], [-e, e, -e], [-e, -e, e], [-e, -e, -e]])
        sample_directions = torch.unsqueeze(sample_directions, 0)
        sample_directions = sample_directions.repeat(bn, int(skel_pnum), 1).cuda()
        sample_centers = torch.repeat_interleave(skel_xyz, 8, dim=1)
        sample_radius = torch.repeat_interleave(skel_radius, 8, dim=1)
        sample_xyz = sample_centers + sample_radius * sample_directions

        cd_sample1 = DF.closest_distance_with_batch(sample_xyz, shape_xyz) / (skel_pnum * 8)
        cd_sample2 = DF.closest_distance_with_batch(shape_xyz, sample_xyz) / (shape_pnum)
        loss_sample = cd_sample1 + cd_sample2

        # point2sphere loss
        skel_xyzr = torch.cat((skel_xyz, skel_radius), 2)
        cd_point2pshere1 = DF.point2sphere_distance_with_batch(shape_xyz, skel_xyzr) / shape_pnum
        cd_point2sphere2 = DF.sphere2point_distance_with_batch(skel_xyzr, shape_xyz) / skel_pnum
        loss_point2sphere = cd_point2pshere1 + cd_point2sphere2

        # radius loss
        loss_radius = - torch.sum(skel_radius) / skel_pnum

        # Laplacian smoothness loss
        loss_smooth = 0
        if lap_reg:
            loss_smooth = self.get_smoothness_loss(skel_xyzr, A) / skel_pnum

        # loss combination
        final_loss = loss_sample + loss_point2sphere * w1 + loss_radius * w2 + loss_smooth * w3

        return final_loss



        # total_loss = F.nll_loss(pred, target)
        # total_loss = 0
        # batch_size = pred.size()[0]
        # Simple Chamfer Distance

        # total_loss = chamfer_distance(pred_[:,:,0:3], target)[0]
        # return total_loss


        # radis only
        # total_loss = 0
        # pred_ = pred.view(batch_size, 20)
        # radis = pred_
        # for i in range(batch_size):
        #     for j in range(20):
        #         center_ = torch.tensor([0.04,0.0,0.0],requires_grad = True).cuda()
        #         center_[1] = 0.075*j -0.7
        #         normal_ = torch.tensor([0.0,1.0,0.0],requires_grad=True).cuda()
        #         radis_ = pred_[i,j]
        #         normal_cz = torch.tensor([-1.0 * normal_[1], normal_[0], 0], requires_grad=True).cuda()
        #         firstPoint = center_ + radis_ * normal_cz
        #         firstPoint = firstPoint - center_
        #         for k in range(0,360,20):
        #             angle = k*2*3.1415926/360
        #             angle = torch.tensor(angle,requires_grad = True).cuda()
        #             point = self.Rotate_Point3d(firstPoint,normal_,angle)
        #             point = point + center_
        #             point = point.unsqueeze(0)
        #             if k == 0:
        #                 points = point
        #             else:
        #                 points = torch.cat((points,point),0)
        #         points = points.unsqueeze(0)
        #         if j == 0:
        #             points_ = points
        #         else:
        #             points_ = torch.cat((points_,points),1)
        #     # points_ = points_.unsqueeze(0)
        #     if i == 0:
        #         points__ = points_
        #     else:
        #         points__ = torch.cat((points__,points_),0)
        #
        # for i in range(batch_size):
        #     disss = 0
        #     for j in range(points__.shape[1]):
        #         pt = points__[i,j,:]
        #         # center_ = torch.tensor([0.04, 0.0, 0.0], requires_grad=True).cuda()
        #         # center_[1] = 0.075 * j - 0.7
        #         # dis = torch.norm(pt-center_)
        #         diss = torch.norm(pt-target[i,:],dim = 1)
        #         dis = torch.min(diss)
        #
        #         dis = dis*dis
        #         disss = disss + dis
        #     total_loss = total_loss + disss
        #
        # return total_loss





        # pred_ = pred.view(batch_size, 10, 7)
        # # genrate the sphere points
        # center = pred_[:,:,0:3]
        # normal = pred_[:,:,3:6]
        # radis = pred_[:,:,6]
        # for i in range(batch_size):
        #     for j in range(10):
        #         center_ = center[i,j,:]
        #         normal_ = normal[i,j,:]
        #         normal_ = normal_/normal_.norm()
        #         radis_ = radis[i,j]
        #         normal_cz = torch.tensor([-1.0*normal_[1],normal_[0],0],requires_grad = True).cuda()
        #         normal_cz = normal_cz/normal_cz.norm()
        #         # disnn = torch.mul(normal_,normal_cz)
        #         firstPoint = center_ + radis_*normal_cz
        #         for k in range(0,360,20):
        #             angle = k*2*3.1415926/360
        #             angle = torch.tensor(angle,requires_grad = True).cuda()
        #             point = self.Rotate_Point3d(firstPoint,normal_,angle)
        #             point = point.unsqueeze(0)
        #             if k == 0:
        #                 points = point
        #             else:
        #                 points = torch.cat((points,point),0)
        #         points = points.unsqueeze(0)
        #         if j == 0:
        #             points_ = points
        #         else:
        #             points_ = torch.cat((points_,points),1)
        #     # points_ = points_.unsqueeze(0)
        #     if i == 0:
        #         points__ = points_
        #     else:
        #         points__ = torch.cat((points__, points_),0)
        # # points__ = points__.view(8,20,36,3)
        # # print(points__.shape)
        # # total_loss = chamfer_distance(points__, target)[0]
        # # compute the one side chamfer distance
        # for i in range(batch_size):
        #     for j in range(points__.shape[1]):
        #         pt = points__[i,j,:]
        #         diss = torch.norm(pt-target[i,:],dim = 1)
        #         dis = torch.min(diss)
        #         dis = dis*dis
        #         dis = dis.unsqueeze(0)
        #         if j == 0:
        #             dis_ = dis
        #         else:
        #             dis_ = torch.cat((dis_,dis),0)
        #     dis_ = dis_.sum()/points__.shape[1]
        #     dis_ = dis_.unsqueeze(0)
        #     if i == 0:
        #         dis__ = dis_
        #     else:
        #         dis__ = torch.cat((dis__,dis_),0)
        # dis__ = dis__.sum()/batch_size
        # # print(dis__)
        # total_loss = dis__
        #
        # # circles cd
        # # points__.view(batch_size,10,18,3)
        # diss =0
        # for i in range(batch_size):
        #     for j in range(0,10):
        #         for k in range(0,10):
        #             pt = points__[i,j*18:j*18+18,:]
        #             pt_ = points__[i,k*18:k*18+18,:]
        #
        #             diss += chamfer_distance(pt.unsqueeze(0),pt_.unsqueeze(0))[0]
        #
        # diss = diss/(10*10*batch_size)
        # # print(diss)
        # total_loss += 0.0001*torch.exp(-1.0*diss)
        # diss = 0
        # for i in range(batch_size):
        #     for j in range(0,10):
        #         for k in range(0,10):
        #             center_ = center[i,j,:]
        #             center__ = center[i,k,:]
        #             dis = torch.norm(center_-center__,dim = 0)
        #             dis = dis*dis
        #             diss += dis
        #
        # diss = diss/(10*10*batch_size)
        # # print(diss)
        # total_loss += 0.0001*torch.exp(-1.0*diss)
        #
        # return total_loss







        ### if we generated circle point directly  目前有问题
        # pred_ = pred.view(batch_size, 20, 3)
        # total_loss = 0.0
        # # total_loss = chamfer_distance(pred_[:, 2:20, 0:3], target)[0]
        # center = pred_[:, 0, 0:3]
        # normal = pred_[:, 1, 0:3]
        # for i in range(batch_size):
        #     for j in range(2, 20):
        #         line = pred_[:, j, 0:3] - center[i, :]
        #         corss = torch.cross(line, normal[i, :]).norm()
        #         total_loss = 10*corss + total_loss
        # return total_loss


