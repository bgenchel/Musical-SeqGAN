# -*- coding:utf-8 -*-

import os
import random
import math

import tqdm

import numpy as np
import torch

class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size

        real_data = self.read_file(real_data_file)
        fake_data = self.read_file(fake_data_file)
        self.data = real_data + fake_data
        self.labels = [1]*len(real_data) + [0]*len(fake_data)
        self.pairs = list(zip(self.data, self.labels))
        self.num_data_pts = len(self.pairs)
        # self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.num_data_pts)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.num_data_pts:
            raise StopIteration
        pairs = [self.pairs[i] for i in range(self.idx, self.idx + self.batch_size)]
        data, labels = zip(*pairs)
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        labels = torch.LongTensor(np.asarray(labels, dtype='int64'))
        self.idx += self.batch_size
        return data, labels

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis

