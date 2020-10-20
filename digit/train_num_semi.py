from __future__ import absolute_import

import torch
import numpy as np 
import cv2
import os
from torch.utils.data import DataLoader
import dataset_num as dataset
import argparse
from PIL import Image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
from torchvision import models
import torch.nn.functional as F
import shutil
import pdb
import sys, time

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, MovingAverage, AverageMeter_Mat
import vgg_n as vgg



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 100]')
parser.add_argument('--print_inter', type=int, default=50, help='print logs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
parser.add_argument('--loss', default='log', help='Loss function [defaultL l2]')
parser.add_argument('--pp', type=float, default=1.0, help='Percentage of target domain data used')
FLAGS = parser.parse_args()


MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
LOSS = FLAGS.loss
PP = FLAGS.pp


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
# pdb.set_trace()

NUM_GPU = len(GPU_INDEX.split(','))
BATCH_SIZE = FLAGS.batch_size * NUM_GPU
print ('Batch Size = %d' %(BATCH_SIZE))
name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR+ '/models')
os.system('cp %s %s' % (name_file, LOG_DIR))
os.system('cp %s %s' % ('*.py', LOG_DIR+ '/models/'))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
print (str(FLAGS))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
st = ' '
log_string(st.join(sys.argv))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
    # if isinstance(m, nn.Linear):
        # nn.init.kaiming_normal(m.weight.data)
        nn.init.normal(m.weight.data, mean=0, std=0.1)
        m.bias.data.zero_()

join_liter = 8000000

def main():
    data_path = '/data/dataset/svhn'

    train_data = dataset.joint_data_loader(data_path,data_path,PP)
    test_data = dataset.joint_data_loader(data_path,data_path, is_training=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=4, num_workers=2, shuffle=False)



    net = vgg.LeNet(14)

    net.apply(weights_init)
    cudnn.benchmark = True

    net.train().cuda()

    criterion = nn.CrossEntropyLoss()


    trainable_list = filter(lambda p: p.requires_grad, net.parameters())
    other_base_list = filter(lambda p: p[0].split('.')[-1]!='bases1', net.named_parameters())
    other_base_list = [x[1] for x in other_base_list if x[1].requires_grad==True]

    named_base_list = filter(lambda p: p[0].split('.')[-1]!='bases0', net.named_parameters())
    base_list = [x[1] for x in named_base_list if x[1].requires_grad==True]


    print('Totolly %d bases for domain1' %(len(base_list)))

    main_list = other_base_list

    optimizer = torch.optim.SGD(net.parameters(), FLAGS.learning_rate, 0.9, weight_decay=0.0001, nesterov=True)

    count = 1
    epoch = 0

    cudnn.benchmark = True

    MOVING_SCALE = 100


    loss_ma0 = MovingAverage(MOVING_SCALE)
    loss_ma1 = MovingAverage(MOVING_SCALE)

    loss_dis_ma = MovingAverage(MOVING_SCALE)
    loss_style_ma = MovingAverage(MOVING_SCALE)
    loss_main_ma = MovingAverage(MOVING_SCALE)
    loss_sim_ma = MovingAverage(MOVING_SCALE)
    #loss_l1 = MovingAverage(100)
    acc_ma_0 = MovingAverage(MOVING_SCALE)
    acc_ma_1 = MovingAverage(MOVING_SCALE)

    test_loss = AverageMeter()
    test_acc = AverageMeter()
    while True:
        if epoch >= MAX_EPOCH: break
        epoch += 1
        log_string('********Epoch %d********' %(epoch))
        for i, data in enumerate(train_loader):
            img0, img1, gt0, gt1 = data
            # img1, img0, gt1, gt0 = data
            # pdb.set_trace()
            count += 1

            imgs_in_0 = Variable(img0).float().cuda()
            gt_in_0 = Variable(gt0).long().cuda()
            imgs_in_1 = Variable(img1).float().cuda()
            gt_in_1 = Variable(gt1).long().cuda()

            pred= net(torch.cat([imgs_in_0, imgs_in_1], 0))
            # pdb.set_trace()
            pred0 = pred[:BATCH_SIZE]
            pred1 = pred[BATCH_SIZE:]

            f0 = net.feature[:BATCH_SIZE]
            f1 = net.feature[BATCH_SIZE:]

            loss0 = criterion(pred0, gt_in_0)
            loss1 = criterion(pred1, gt_in_1)

            loss_all = loss0 + loss1


            acc_this_0 = accuracy(pred0, gt_in_0)
            acc_ma_0.update(acc_this_0[0].item())

            acc_this_1 = accuracy(pred1, gt_in_1)
            acc_ma_1.update(acc_this_1[0].item())


            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_ma0.update(loss0.item())
            loss_ma1.update(loss1.item())


            if count % FLAGS.print_inter == 0:
                log_string('[Current iter %d, accuracy0 is %3.2f, accuracy1 is %3.2f, \
loss0 is %2.6f, loss1 is %2.6f, lr: %f]' 
                    %(count, acc_ma_0.avg, acc_ma_1.avg, \
                        loss_ma0.avg, loss_ma1.avg, \
                        optimizer.param_groups[0]['lr']))
            if count % 500 == 0:
               validation(test_loader, net, criterion, count, epoch, test_loss, test_acc)
            if count % 1000 == 0:
                torch.save(net.state_dict(), './' + LOG_DIR + '/' + 'model.pth')
            if count % DECAY_STEP == 0 and optimizer.param_groups[0]['lr'] >= (BASE_LEARNING_RATE / 10.0):
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10.0
                optimizer_base1.param_groups[0]['lr'] = optimizer_base1.param_groups[0]['lr']/10.0
                optimizer_dis.param_groups[0]['lr'] = optimizer_dis.param_groups[0]['lr']/10.0

    log_string('Training reaches maximum epoch.')


def validation(dataloader, net, criterion, count, epoch, test_loss, test_acc):
    log_string('Validating at Epoch %d, iter %d ---------------------------------' %(epoch, count))
    net.eval()
    test_loss.reset()
    # test_acc.reset()
    acc_0 = AverageMeter()
    acc_1 = AverageMeter()
    time_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img0, img1, gt0, gt1 = data

            imgs_in_0 = Variable(img0).float().cuda()
            gt_in_0 = Variable(gt0).long().cuda()
            imgs_in_1= Variable(img1).float().cuda()
            gt_in_1 = Variable(gt1).long().cuda()
            # pdb.set_trace()

            time1 = time.time()
            pred = net(torch.cat([imgs_in_0, imgs_in_1], 0))
            time2 = time.time()

            pred0 = pred[:4]
            pred1 = pred[4:]

            time_list.append(time2-time1)

            acc_this_0 = accuracy(pred0, gt_in_0)
            acc_this_1 = accuracy(pred1, gt_in_1)
            acc_0.update(acc_this_0[0].item())
            acc_1.update(acc_this_1[0].item())


            loss0 = criterion(pred0, gt_in_0)
            loss1 = criterion(pred1, gt_in_1)
            loss = loss0 + loss1

            test_loss.update(loss.item())

    log_string('---------------------------------------------')
    log_string('[Test loss is %2.5f, acc1 is %3.2f, acc2 is %3.2f, speed at %0.5f s/frame.]'
        %(test_loss.avg, acc_0.avg, acc_1.avg, np.array(time_list).mean()))
    log_string('---------------------------------------------')

    log_string('Validating Done, going back to Training')

    net.train()



if __name__ == "__main__":
    main()
