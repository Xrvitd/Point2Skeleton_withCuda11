"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

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
    parser.add_argument('--num_category', default=70, type=int, help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    # parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
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

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
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
    mypc1 = o3d.io.read_point_cloud(data_path+'1024pts.xyz', format='xyzn')
    mypc1 = np.asarray(mypc1.points)
    mypc2 = o3d.io.read_point_cloud(data_path+'1400pts.xyz', format='xyzn')
    mypc2 = np.asarray(mypc2.points)
    mypc3 = o3d.io.read_point_cloud(data_path+'4200pts.xyz', format='xyzn')
    mypc3 = np.asarray(mypc3.points)
    mypc4 = o3d.io.read_point_cloud(data_path+'7000pts.xyz', format='xyzn')
    mypc4 = np.asarray(mypc4.points)



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
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # try:
    #     checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0
    log_string('Do not use pretrain model...')
    start_epoch = 0

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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_instance_acc = 999999
    '''TRANING'''
    batch_size = 4
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []


        scheduler.step()
        print('learning rate: %f' % scheduler.get_lr()[0])

        # for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        for batch_id in tqdm(range(4), smoothing=0.9):
            optimizer.zero_grad()
            global points, target

            points = np.array([mypc3,mypc3,mypc3,mypc3])
            target = np.array([mypc3,mypc3,mypc3,mypc3])
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            target = torch.Tensor(target) #.add.d

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target, trans_feat)
            # pred_choice = pred.data.max(1)[1]

            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()
            global_step += 1

        # train_instance_acc = np.mean(mean_correct)
        # log_string('Train Instance Accuracy: %f' % train_instance_acc)


        '''TESTING'''
        loss = criterion(pred, target, trans_feat)
        print('loss: %f' % loss)
        with torch.no_grad():
            # instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (loss <= best_instance_acc):
                best_instance_acc = loss
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
                # for branch_idx in range(8):
                #     with open(str(checkpoints_dir)+'/best_Points%d.xyz' % branch_idx, "w") as f:
                #         for i in range(20):
                #             [x, y, z, nx, ny, nz, r] = pred[branch_idx][i * 7:i * 7 + 7]
                #             x = x.cpu().numpy()
                #             y = y.cpu().numpy()
                #             z = z.cpu().numpy()
                #             f.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
                #         f.write("\n")
                pred_ = pred.view(batch_size, 10, 7)
                center = pred_[:, :, 0:3]
                normal = pred_[:, :, 3:6]
                radis = pred_[:, :, 6]
                for i in range(batch_size):
                    for j in range(10):
                        center_ = center[i, j, :]
                        normal_ = normal[i, j, :]
                        normal_ = normal_ / normal_.norm()
                        radis_ = radis[i, j]
                        normal_cz = torch.tensor([-1.0 * normal_[1], normal_[0], 0]).cuda()
                        normal_cz = normal_cz / normal_cz.norm()
                        # disnn = torch.mul(normal_,normal_cz)
                        firstPoint = center_ + radis_ * normal_cz
                        for k in range(0, 360, 10):
                            angle = k * 2 * 3.1415926 / 360
                            angle = torch.tensor(angle).cuda()
                            point = Rotate_Point3d(firstPoint, normal_, angle)
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
                points__ = points__.cpu().numpy()
                for i in range(batch_size):
                    with open(str(checkpoints_dir) + '/best_Points%d.xyz' % i, "w") as f:
                        for j in range(points__.shape[1]):
                            [x, y, z] = points__[i][j]
                            f.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
                        f.write("\n")

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



            global_epoch += 1
            log_string('Best Loss: %f, Class Loss: %f' % (best_instance_acc, loss))



    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
