from __future__ import absolute_import

import numpy as np
import cv2
from PIL import Image
import os
import pdb
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.io as sio
import math
# from pose.utils.osutils import *
# from pose.utils.imutils import *
# from pose.utils.transforms import *
# from pose.utils.evaluation import *
import struct
import skimage.transform as st
import pickle as pkl

color_list = np.array(np.random.random((40, 3)) * 255, int)



def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels




class mnist_data_loader(data.DataLoader):
    def __init__(self, path, resize=32, is_training=True):
        self.is_training = is_training
        if is_training:
            data, label = load_mnist(path, 'train')
        else:
            data, label = load_mnist(path, 't10k')

        self.resize = resize
        self.data = np.reshape(data, (-1, 28, 28))
        self.label = label
    def __getitem__(self, index):

        img = self.data[index]
        img = np.tile(np.expand_dims(img, -1), (1, 1, 3))

        img = cv2.resize(img, (32, 32))
        #cv2.imwrite('img%d.png' %(self.label[index]), img)
        img = np.array(img, np.float32)
        img = ((img/255.0)-0.5)/0.5
        img = np.transpose(img, (2,0,1))
        label = self.label[index]

        return img, label


    def __len__(self):
        return len(self.label)

class svhn_data_loader(data.DataLoader):
    def __init__(self, path, max_data=-1, is_training=True):
        if is_training:
            data = sio.loadmat('train_32x32.mat')
        else:
            data = sio.loadmat('test_32x32.mat')


        self.imgs = np.transpose(data['X'], (3,0,1,2))
        self.labels = data['y']

        if max_data > 0:
            self.imgs = self.imgs[:max_data]
            self.labels = self.labels[:max_data]

    def __getitem__(self, index):
        img = self.imgs[index]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
        img = np.array(np.tile(np.expand_dims(img, -1), (1,1,3)), np.float32)
        img = ((img/255.0)-0.5)/0.5
        img = np.transpose(img, (2,0,1))    
        label = self.labels[index,0]

        return img, label

    def __len__(self):
        return len(self.labels)



class joint_data_loader(data.DataLoader):
    def __init__(self, path_mnist='./', path_to_svhn='./', pp=1.0 ,is_training=True):
        if is_training:
            data_mnist, label_mnist = load_mnist(path_mnist, 'train')
            data_svhn = sio.loadmat(os.path.join(path_to_svhn, 'train_32x32.mat'))
        else:
            data_mnist, label_mnist = load_mnist(path_mnist, 't10k')
            data_svhn = sio.loadmat(os.path.join(path_to_svhn, 'test_32x32.mat'))


        self.imgs_svhn = np.transpose(data_svhn['X'], (3,0,1,2))
        self.labels_svhn = data_svhn['y']
        if is_training and pp!=1.0:
            if pp < 1:
                l = int(pp * len(self.imgs_svhn))
            else:
                l = int(pp)
            self.imgs_svhn = self.imgs_svhn[:l]
            self.labels_svhn = self.labels_svhn[:l]

        self.imgs_mnist = np.reshape(data_mnist, (-1, 28, 28))
        self.labels_mnist = label_mnist

        self.is_training = is_training


    def __getitem__(self, index):
        index_mnist = np.random.randint(len(self.labels_mnist))
        index_svhn = np.random.randint(len(self.labels_svhn))
        # print(index_mnist)
        # print(index_svhn)

        #svhn
        img_svhn = self.imgs_svhn[index_svhn]
        img_svhn = cv2.cvtColor(img_svhn, cv2.COLOR_RGB2GRAY)
        img_svhn = np.array(np.tile(np.expand_dims(img_svhn, -1), (1,1,3)), np.float32)
        img_svhn = ((img_svhn/255.0)-0.5)/0.5
        img_svhn = np.transpose(img_svhn, (2,0,1))  

        label_svhn = self.labels_svhn[index_svhn, 0]
        if label_svhn == 10:
            label_svhn -= 10

        #mnist
        img_mnist = self.imgs_mnist[index_mnist]
        img_mnist = np.tile(np.expand_dims(img_mnist, -1), (1, 1, 3))

        img_mnist = cv2.resize(img_mnist, (32, 32))
        #cv2.imwrite('img%d.png' %(self.label[index]), img)
        img_mnist = np.array(img_mnist, np.float32)
        img_mnist = ((img_mnist/255.0)-0.5)/0.5
        img_mnist = np.transpose(img_mnist, (2,0,1))
        label_mnist = self.labels_mnist[index_mnist]

        return img_mnist, img_svhn, label_mnist, label_svhn

    def __len__(self):
        if self.is_training:
            return 1000000
        else:
            return 1000

class joint_data_loader_m(data.DataLoader):
    def __init__(self, path_mnist='./', path_to_svhn='./', pp=1.0 ,is_training=True):

        f = open('mnistm_data.pkl', 'rb')
        data_m = pkl.load(f, encoding='latin1')
        f.close()
        if is_training:
            data_mnist, label_mnist = load_mnist(path_mnist, 'train')
            # data_svhn = sio.loadmat(os.path.join(path_to_svhn, 'train_32x32.mat'))
            data_mnistm = np.concatenate([data_m['valid'], data_m['train']], 0)
            label_mnistm = label_mnist.copy()
        else:
            data_mnist, label_mnist = load_mnist(path_mnist, 't10k')
            data_mnistm = data_m['test']
            label_mnistm = label_mnist.copy()
            # data_svhn = sio.loadmat(os.path.join(path_to_svhn, 'test_32x32.mat'))


        self.imgs_mnistm = data_mnistm
        self.labels_mnistm = label_mnistm
        if is_training and pp!=1.0:
            if pp < 1:
                l = int(pp * len(self.imgs_mnistm))
            else:
                l = int(pp)
            self.imgs_mnistm = self.imgs_mnistm[:l]
            self.labels_mnistm = self.labels_mnistm[:l]

        self.imgs_mnist = np.reshape(data_mnist, (-1, 28, 28))
        self.labels_mnist = label_mnist

        self.is_training = is_training


    def __getitem__(self, index):
        index_mnist = np.random.randint(len(self.labels_mnist))
        index_mnistm = np.random.randint(len(self.labels_mnistm))
        # print(index_mnist)
        # print(index_svhn)

        #svhn
        img_mnistm = self.imgs_mnistm[index_mnistm]
        img_mnistm = cv2.resize(img_mnistm, (32, 32))
        # img_svhn = cv2.cvtColor(img_svhn, cv2.COLOR_RGB2GRAY)
        # img_svhn = np.array(np.tile(np.expand_dims(img_svhn, -1), (1,1,3)), np.float32)
        img_mnistm = ((img_mnistm/255.0)-0.5)/0.5
        img_mnistm = np.transpose(img_mnistm, (2,0,1))  

        label_mnistm = self.labels_mnistm[index_mnistm]
        # if label_svhn == 10:
        #     label_svhn -= 10

        #mnist
        img_mnist = self.imgs_mnist[index_mnist]
        img_mnist = np.tile(np.expand_dims(img_mnist, -1), (1, 1, 3))

        img_mnist = cv2.resize(img_mnist, (32, 32))
        #cv2.imwrite('img%d.png' %(self.label[index]), img)
        img_mnist = np.array(img_mnist, np.float32)
        img_mnist = ((img_mnist/255.0)-0.5)/0.5
        img_mnist = np.transpose(img_mnist, (2,0,1))
        label_mnist = self.labels_mnist[index_mnist]

        return img_mnist, img_mnistm, label_mnist, label_mnistm

    def __len__(self):
        if self.is_training:
            return 1000000
        else:
            return 1000


if __name__ == "__main__":
    dataset = joint_data_loader()
    train_loader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True, pin_memory=True)
    # for i, data in enumerate(train_loader):
    #     imgs, gt = data
    #     # print gt.shape
    #     gt = np.array(gt[0])
    #     img = imgs[0].numpy()
    #     # print img
    #     img = np.transpose(img, (1,2,0)).copy()
    #     # print img.shape, gt
    #     # print img
    #     img = img * 0.5 + 0.5

    #     img = img * 255.0
    #     img = np.array(img, np.uint8)
    #     # print img.min()

    #     # pdb.set_trace()

    #     draw_img(img, gt)

    #     cv2.imshow('img', img)
    #     cv2.waitKey()

    for i, data in enumerate(train_loader):
        a,b,c,d = data
        # pdb.set_trace()
        img_a = a[0]
        img_a = (img_a*0.5+0.5)*255
        img_a = img_a.numpy().transpose(1,2,0)
        img_a = np.array(img_a, np.uint8)
        cv2.imwrite('a.jpg', img_a)

        img_b = b[0]
        img_b = (img_b*0.5+0.5)*255
        img_b = img_b.numpy().transpose(1,2,0)
        img_b = np.array(img_b, np.uint8)
        cv2.imwrite('b.jpg', img_b)
