import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def Rotate_Point3d(self, point, aix, angle):
        # point: [x,y,z]
        # aix: [x,y,z]
        # angle: radian
        # return: [x,y,z]
        aix = aix / aix.norm()
        x = point[0]
        y = point[1]
        z = point[2]
        u = aix[0]
        v = aix[1]
        w = aix[2]
        cos = torch.cos(angle)  # 弧度制
        sin = torch.sin(angle)
        R = torch.tensor([[u * u + (1 - u * u) * cos, u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v * sin],
                          [u * v * (1 - cos) + w * sin, v * v + (1 - v * v) * cos, v * w * (1 - cos) - u * sin],
                          [u * w * (1 - cos) - v * sin, v * w * (1 - cos) + u * sin, w * w + (1 - w * w) * cos]],
                         requires_grad=True).cuda()
        point = torch.tensor([x, y, z], requires_grad=True).cuda()
        point = torch.matmul(R, point)
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
        center = pred_[:, :, 0:3]
        normal = pred_[:, :, 3:6]
        radis = pred_[:, :, 6]
        for i in range(batch_size):
            for j in range(10):
                center_ = center[i, j, :]
                normal_ = normal[i, j, :]
                normal_ = normal_ / normal_.norm()
                radis_ = radis[i, j]
                normal_cz = torch.tensor([-1.0 * normal_[1], normal_[0], 0], requires_grad=True).cuda()
                normal_cz = normal_cz / normal_cz.norm()
                # disnn = torch.mul(normal_,normal_cz)
                firstPoint = center_ + radis_ * normal_cz
                for k in range(0, 360, 20):
                    angle = k * 2 * 3.1415926 / 360
                    angle = torch.tensor(angle, requires_grad=True).cuda()
                    point = self.Rotate_Point3d(firstPoint, normal_, angle)
                    point = point.unsqueeze(0)
                    if k == 0:
                        points = point
                    else:
                        points = torch.cat((points, point), 0)
                points = points.unsqueeze(0)
                if j == 0:
                    points_ = points
                else:
                    points_ = torch.cat((points_, points), 1)
            # points_ = points_.unsqueeze(0)
            if i == 0:
                points__ = points_
            else:
                points__ = torch.cat((points__, points_), 0)
        # points__ = points__.view(8,20,36,3)
        # print(points__.shape)
        # total_loss = chamfer_distance(points__, target)[0]
        # compute the one side chamfer distance
        for i in range(batch_size):
            for j in range(points__.shape[1]):
                pt = points__[i, j, :]
                diss = torch.norm(pt - target[i, :], dim=1)
                dis = torch.min(diss)
                dis = dis * dis
                dis = dis.unsqueeze(0)
                if j == 0:
                    dis_ = dis
                else:
                    dis_ = torch.cat((dis_, dis), 0)
            dis_ = dis_.sum() / points__.shape[1]
            dis_ = dis_.unsqueeze(0)
            if i == 0:
                dis__ = dis_
            else:
                dis__ = torch.cat((dis__, dis_), 0)
        dis__ = dis__.sum() / batch_size
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


if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))