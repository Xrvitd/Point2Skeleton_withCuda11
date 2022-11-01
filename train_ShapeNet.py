"""
Author: Benny
Date: Nov 2019
"""

import os
import sys

import numpy
import torch
import numpy as np
import open3d as o3d

import random
import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import data_utils.FileRW as rw
from data_utils.MyDataLoader import PCDataset
from torch.utils.data import Dataset, DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=20, type=int, help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=60):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc



def Rotate_Point3d(point,aix,angle):
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
                     [u*w*(1-cos)-v*sin,v*w*(1-cos)+u*sin,w*w+(1-w*w)*cos]]).cuda()
    point = torch.tensor([x,y,z]).cuda()
    point = torch.matmul(R,point)
    return point

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = 4
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('ShapeNet')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = 'data/modelnet40_normal_resampled/'
    #
    # train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    data_path = 'data/MyPoints/'
    # train_dataset = torch.utils.data.DataLoader( , batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    pc_list_file = 'data/data-split/little-train1.txt'
    data_root = 'data/pointclouds/'
    pc_list = rw.load_data_id(pc_list_file)
    train_data = PCDataset(pc_list, data_root, args.num_point)
    train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True, drop_last=True)
    pc_list_file = 'data/data-split/little-test1.txt'
    data_root = 'data/pointclouds/'
    pc_list = rw.load_data_id(pc_list_file)
    test_data = PCDataset(pc_list, data_root, args.num_point)
    test_loader = DataLoader(dataset=test_data, batch_size = 1, shuffle=False, drop_last=False)







    # trainDataLoader = torch.utils.data.DataLoader(Mypcs, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    #
    # testDataLoader = torch.utils.data.DataLoader(Mypcs, batch_size=args.batch_size, shuffle=False, num_workers=0)


    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    # classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    criterion_pre = model.get_loss_pre()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        criterion_pre = criterion_pre.cuda()


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    start_epoch = 0
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model_ShapeNet.pth')
        start_epoch = 50
        # start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    # log_string('Do not use pretrain model...')
    # start_epoch = 0
    for params in optimizer.param_groups:
        params['lr'] = args.learning_rate




    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_instance_acc = 999999
    best_instance_acc_pre = 999999
    '''TRANING'''
    batch_size = 4
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []

        scheduler.step()
        print('learning rate: %f' % scheduler.get_lr()[0])

        loss_batch = 0
        # for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        # for batch_id in tqdm(range(10), smoothing=0.9):
        for k, batch_data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.8):
            optimizer.zero_grad()

            batch_id, batch_pc = batch_data
            batch_id = batch_id
            batch_pc = batch_pc.cuda().float()
            # global points, target

            points = batch_pc[:,:,0:3]
            target = batch_pc
            # target = np.array(mypcs[batch_id*4:batch_id*4+4])
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            target = torch.Tensor(target) #.add.d

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # pred, trans_feat = classifier(points)
            # loss = criterion(pred, target, trans_feat)
            skel_xyz, skel_r, shape_cmb_features,skel_nori ,weights,l3_xyz,l3_normals= classifier(points)

            if epoch<20:
                loss_pre = criterion_pre(target,skel_xyz)
                loss_batch += loss_pre.item()
                optimizer.zero_grad()
                loss_pre.backward()
                optimizer.step()
                global_step += 1
                # log_string('loss_pre: %f' % (loss_pre.item()))
            else:
                loss = criterion(skel_xyz, skel_r, shape_cmb_features, skel_nori,
                                 weights,l3_xyz,l3_normals, target, None, 0.3, 0.4,0, 0.01)
                loss_batch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

        loss_batch = loss_batch/batch_size
                # log_string('loss: %f' % (loss.item()))

        '''TESTING'''

        if epoch < 20:
            with torch.no_grad():
                if (loss_batch <= best_instance_acc_pre):
                    best_instance_acc_pre = loss_batch
                    best_epoch = epoch + 1
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'best_loss': best_instance_acc_pre,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
            print('Pretraining loss_pre: %f' % loss_batch, 'Bestloss: %f' % best_instance_acc_pre)
        else:
            with torch.no_grad():
                if (loss_batch <= best_instance_acc):
                    best_instance_acc = loss_batch
                    best_epoch = epoch + 1
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'best_loss': best_instance_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    # skel_nori = skel_nori / torch.norm(skel_nori, dim=2, keepdim=True)
                    for branch_idx in range(batch_size):
                        with open(str(checkpoints_dir)+'/best_Points%d.xyz' % branch_idx, "w") as f:
                            for i in range(len(skel_xyz[branch_idx])):
                                f.write("%f %f %f " % (skel_xyz[branch_idx][i][0], skel_xyz[branch_idx][i][1], skel_xyz[branch_idx][i][2]))
                                f.write("%f %f %f\n" % (skel_nori[branch_idx][i][0], skel_nori[branch_idx][i][1], skel_nori[branch_idx][i][2]))
                        with open(str(checkpoints_dir)+'/best_Radii%d.xyz' % branch_idx, "w") as f:
                            for i in range(len(skel_r[branch_idx])):
                                f.write("%f\n" % skel_r[branch_idx][i])
                        with open(str(checkpoints_dir)+'/best_l3points%d.xyz' % branch_idx, "w") as f:
                            for i in range(len(l3_xyz[branch_idx])):
                                f.write("%f %f %f " % (l3_xyz[branch_idx][i][0], l3_xyz[branch_idx][i][1], l3_xyz[branch_idx][i][2]))
                                f.write("%f %f %f\n" % (l3_normals[branch_idx][i][0], l3_normals[branch_idx][i][1], l3_normals[branch_idx][i][2]))

            print('Skeletal training loss: %f' % loss_batch, 'Bestloss: %f' % best_instance_acc)

        global_epoch += 1




                #         for i in range(20):
                #             [x, y, z, nx, ny, nz, r] = pred[branch_idx][i * 7:i * 7 + 7]
                #             x = x.cpu().numpy()
                #             y = y.cpu().numpy()
                #             z = z.cpu().numpy()
                #             f.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
                #         f.write("\n")
                # pred_ = pred.view(batch_size, 20)
                # # center = pred_[:, :, 0:3]
                # # normal = pred_[:, :, 3:6]
                # # radis = pred_[:, :, 6]
                # for i in range(batch_size):
                #     for j in range(20):
                #         # center_ = center[i, j, :]
                #         # normal_ = normal[i, j, :]
                #         # normal_ = normal_ / normal_.norm()
                #         # radis_ = radis[i, j]
                #         center_ = torch.tensor([0.04, 0, 0], requires_grad=True).cuda()
                #         center_[1] = 0.075 * j - 0.7
                #         normal_ = torch.tensor([0.0, 1, 0], requires_grad=True).cuda()
                #         radis_ = pred_[i, j]
                #
                #         normal_cz = torch.tensor([-1.0 * normal_[1], normal_[0], 0]).cuda()
                #         normal_cz = normal_cz / normal_cz.norm()
                #         # disnn = torch.mul(normal_,normal_cz)
                #         firstPoint = center_ + radis_ * normal_cz
                #         firstPoint = firstPoint - center_
                #         for k in range(0, 360, 10):
                #             angle = k * 2 * 3.1415926 / 360
                #             angle = torch.tensor(angle).cuda()
                #             point = Rotate_Point3d(firstPoint, normal_, angle)
                #             point = point + center_
                #             point = point.unsqueeze(0)
                #             if k == 0:
                #                 points = point
                #             else:
                #                 points = torch.cat((points, point), 0)
                #         points = points.unsqueeze(0)
                #         if j == 0:
                #             points_ = points
                #         else:
                #             points_ = torch.cat((points_, points), 1)
                #     # points_ = points_.unsqueeze(0)
                #     if i == 0:
                #         points__ = points_
                #     else:
                #         points__ = torch.cat((points__, points_), 0)
                # points__ = points__.cpu().numpy()
                # for i in range(batch_size):
                #     with open(str(checkpoints_dir) + '/best_Points%d.xyz' % i, "w") as f:
                #         for j in range(points__.shape[1]):
                #             [x, y, z] = points__[i][j]
                #             f.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
                #         f.write("\n")

            # pred_ = pred.view(batch_size, 20, 3)
            # pred_ = pred_.cpu().numpy()
            #
            # for i in range(batch_size):
            #     with open(str(checkpoints_dir) + '/best_Points%d.xyz' % i, "w") as f:
            #         for j in range(2, 20):
            #             [x,y,z] = pred_[i][j]
            #             f.write(str(x) + ' ' + str(y) + ' ' + str(z)+ '\n')
            #         center = pred_[i, 0, 0:3]
            #         f.write(str(center[0]) + ' ' + str(center[1]) + ' ' + str(center[2]))
            #         f.write("\n")







    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
