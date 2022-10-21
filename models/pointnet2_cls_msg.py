import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction



class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc0 = nn.Linear(12840, 1024) #全连接层

        self.fc1 = nn.Linear(1024, 512)
        self.fc1_ = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_ = nn.Linear(256, 256)
        self.fc2__ = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
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

        # x = ((self.fc1(x)))
        # x = ((self.fc2(x)))
        # x = self.fc3(x)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)

        # x = self.fc0(xyz.reshape(1, -1))
        x = ((self.fc1(x)))
        x = self.fc1_(x)
        x = ((self.fc2(x)))
        x = ((self.fc2_(x)))
        x = ((self.fc2__(x)))
        # # x = ((self.fc2_(x)))
        x = self.fc3(x)
        # x = self.fc0(xyz.view(3076))
        # x = self.bn1(self.fc1(x))
        # x = self.bn2(self.fc2(x))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

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

    def forward(self, pred, target, trans_feat):
        # total_loss = F.nll_loss(pred, target)
        # total_loss = 0
        batch_size = pred.size()[0]
        # Simple Chamfer Distance

        # total_loss = chamfer_distance(pred_[:,:,0:3], target)[0]
        # return total_loss


        # radis only
        total_loss = 0
        pred_ = pred.view(batch_size, 10)
        radis = pred_
        for i in range(batch_size):
            for j in range(10):
                center_ = torch.tensor([0.04,0.0,0.0],requires_grad = True).cuda()
                center_[1] = 0.15*j -0.7
                normal_ = torch.tensor([0.0,1.0,0.0],requires_grad=True).cuda()
                radis_ = pred_[i,j]
                normal_cz = torch.tensor([-1.0 * normal_[1], normal_[0], 0], requires_grad=True).cuda()
                firstPoint = center_ + radis_ * normal_cz
                firstPoint = firstPoint - center_
                for k in range(0,360,10):
                    angle = k*2*3.1415926/360
                    angle = torch.tensor(angle,requires_grad = True).cuda()
                    point = self.Rotate_Point3d(firstPoint,normal_,angle)
                    point = point + center_
                    point = point.unsqueeze(0)
                    if k == 0:
                        points = point
                    else:
                        points = torch.cat((points,point),0)
                points = points.unsqueeze(0)
                if j == 0:
                    points_ = points
                else:
                    points_ = torch.cat((points_,points),1)
            # points_ = points_.unsqueeze(0)
            if i == 0:
                points__ = points_
            else:
                points__ = torch.cat((points__,points_),0)

        for i in range(batch_size):
            for j in range(points__.shape[1]):
                pt = points__[i,j,:]
                diss = torch.norm(pt-target[i,:],dim = 1)
                dis = torch.min(diss)
                dis = dis*dis
                dis = dis.unsqueeze(0)
                if j == 0:
                    dis_ = dis
                else:
                    dis_ = torch.cat((dis_,dis),0)
            dis_ = dis_.sum()/points__.shape[1]
            dis_ = dis_.unsqueeze(0)
            if i == 0:
                dis__ = dis_
            else:
                dis__ = torch.cat((dis__,dis_),0)
        dis__ = dis__.sum()/batch_size
        # print(dis__)
        total_loss = dis__
        return total_loss





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


