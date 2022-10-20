import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_ = nn.Linear(256, 256)
        self.fc2__ = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.01)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.dropout(self.fc2(x))))

        x = F.relu((self.fc1(x)))
        x = F.relu((self.dropout(self.fc2(x))))
        x = F.relu((self.dropout(self.fc2_(x))))
        x = F.relu((self.dropout(self.fc2__(x))))


        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

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

        pred_ = pred.view(batch_size, 10, 7)
        # genrate the sphere points
        center = pred_[:,:,0:3]
        normal = pred_[:,:,3:6]
        radis = pred_[:,:,6]
        for i in range(batch_size):
            for j in range(10):
                center_ = center[i,j,:]
                normal_ = normal[i,j,:]
                normal_ = normal_/normal_.norm()
                radis_ = radis[i,j]
                normal_cz = torch.tensor([-1.0*normal_[1],normal_[0],0],requires_grad = True).cuda()
                normal_cz = normal_cz/normal_cz.norm()
                # disnn = torch.mul(normal_,normal_cz)
                firstPoint = center_ + radis_*normal_cz
                for k in range(0,360,20):
                    angle = k*2*3.1415926/360
                    angle = torch.tensor(angle,requires_grad = True).cuda()
                    point = self.Rotate_Point3d(firstPoint,normal_,angle)
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
                points__ = torch.cat((points__, points_),0)
        # points__ = points__.view(8,20,36,3)
        # print(points__.shape)
        # total_loss = chamfer_distance(points__, target)[0]
        # compute the one side chamfer distance
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

