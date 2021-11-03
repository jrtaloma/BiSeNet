#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.base_dataset import BaseDataset



class T1(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(T1, self).__init__(dataroot, annpath, trans_func, mode)
        self.n_cats = 2
        self.lb_ignore = 255

        ## label mapping
        labels = np.arange(self.n_cats)
        self.lb_map = np.arange(256)
        for ind in labels:
            self.lb_map[ind] = labels.index(ind)

        self.to_tensor = T.ToTensor(
            mean=(0.5, 0.5, 0.5), # t1, grayscale
            std=(0.5, 0.5, 0.5),
        )
