import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import io
import cv2
import numpy as np
import os, sys, pdb
import glob
from PIL import Image
from colour.plotting import *
from colour_demosaicing import (mosaicing_CFA_Bayer,
                                demosaicing_CFA_Bayer_bilinear)


class NIRVIS(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.ids = os.listdir(root_dir)
        self.subset = subset
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        label = int(idx)
        NIR_list = os.listdir(os.path.join(self.root_dir, self.ids[idx], 'NIR_face'))
        NIR_list.sort()

        VIS_list = os.listdir(os.path.join(self.root_dir, self.ids[idx], 'VIS_face'))
        VIS_list.sort()

        if self.subset == 'train':
            if len(VIS_list) == 1:
                VIS_im_name = VIS_list[0]
            else:
                VIS_im_name = np.random.choice(VIS_list[:-1])
            if len(NIR_list) == 1:
                NIR_im_name = NIR_list[0]
            else:
                NIR_im_name = np.random.choice(NIR_list[:-1])
            
        else:
            NIR_im_name = NIR_list[-1]
            VIS_im_name = VIS_list[-1]

        nir = self.transform(Image.open(os.path.join(self.root_dir, self.ids[idx], 'NIR_face', NIR_im_name)))
        vis = self.transform(Image.open(os.path.join(self.root_dir, self.ids[idx], 'VIS_face', VIS_im_name)))

        return nir, vis, label
