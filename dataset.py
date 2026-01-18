#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import json
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import numpy as np

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class TestDataset(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDataset(Dataset):
    def __init__(self, user_count, item_count, pos_sampling, neg_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.pos_sampling = pos_sampling
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.neg_count = neg_count

    def __getitem__(self, idx):
        total = []
        pos = self.pos_sampling[idx]
        u_id = pos[0]
        total.append(pos)

        all_inter = self.behavior_dict['all'].get(str(u_id), [])
        for _ in range(self.neg_count):
            neg_item = random.randint(1, self.item_count)
            while neg_item in all_inter:
                neg_item = random.randint(1, self.item_count)
            neg_sample = list(pos)
            neg_sample[1] = neg_item
            neg_sample[-1] = 0
            total.append(neg_sample)

        buy_inter = self.behavior_dict[self.behaviors[-1]].get(str(u_id), [])
        if not buy_inter:
            signal = [0, 0, 0, 0]
        else:
            p_item = random.choice(buy_inter)
            n_item = random.randint(1, self.item_count)
            while n_item in all_inter:
                n_item = random.randint(1, self.item_count)
            signal = [u_id, p_item, n_item, 0]
        total.append(signal)
        
        return np.array(total, dtype=np.int32)

    def __len__(self):
        return len(self.pos_sampling)


class DataSet:
    def __init__(self, args):
        self.behaviors = args.behaviors
        self.path = args.data_path
        self.loss_type = getattr(args, 'loss_type', 'bpr')
        self.neg_count = args.neg_count

        self.__get_count()
        self.__get_pos_sampling()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()
        self.__get_validation_behavior_dict()
        self.__get_test_behavior_dict()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
        self.user_count = count['user']
        self.item_count = count['item']

    def __get_pos_sampling(self):
        with open(os.path.join(self.path, 'pos_sampling.txt'), encoding='utf-8') as f:
            data = f.readlines()
        self.pos_sampling = [[int(x) for x in line.strip().split()] for line in data]

    def __get_behavior_items(self):
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, f'{behavior}_dict.txt'), encoding='utf-8') as f:
                self.train_behavior_dict[behavior] = json.load(f)
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            self.train_behavior_dict['all'] = json.load(f)

    def __get_test_dict(self):
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            self.test_interacts = json.load(f)

    def __get_validation_dict(self):
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            self.validation_interacts = json.load(f)

    def __get_validation_behavior_dict(self):
        self.validation_behavior_dict = {}
        buy_behavior = self.behaviors[-1]
        validation_buy_path = os.path.join(self.path, f'validation_{buy_behavior}_dict.txt')
        
        if os.path.exists(validation_buy_path):
            try:
                with open(validation_buy_path, encoding='utf-8') as f:
                    self.validation_behavior_dict[buy_behavior] = json.load(f)
            except Exception:
                self.validation_behavior_dict[buy_behavior] = {}
        else:
            self.validation_behavior_dict[buy_behavior] = {}
        
        for behavior in self.behaviors[:-1]:
            self.validation_behavior_dict[behavior] = {}

    def __get_test_behavior_dict(self):
        self.test_behavior_dict = {}
        buy_behavior = self.behaviors[-1]
        test_buy_path = os.path.join(self.path, f'test_{buy_behavior}_dict.txt')
        
        if os.path.exists(test_buy_path):
            try:
                with open(test_buy_path, encoding='utf-8') as f:
                    self.test_behavior_dict[buy_behavior] = json.load(f)
            except Exception:
                self.test_behavior_dict[buy_behavior] = {}
        else:
            self.test_behavior_dict[buy_behavior] = {}
        
        for behavior in self.behaviors[:-1]:
            self.test_behavior_dict[behavior] = {}

    def __get_sparse_interact_dict(self):
        self.inter_matrix = []
        self.user_item_inter_set = []
        all_row, all_col = [], []

        for behavior in self.behaviors:
            row, col = [], []
            with open(os.path.join(self.path, f'{behavior}.txt'), encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    row.append(int(parts[0]))
                    col.append(int(parts[1]))
            inter_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)),
                                         shape=[self.user_count + 1, self.item_count + 1])
            self.inter_matrix.append(inter_matrix)
            self.user_item_inter_set.append([list(r.indices) for r in inter_matrix.tocsr()])
            all_row.extend(row)
            all_col.extend(col)

        all_edges = list(set(zip(all_row, all_col)))
        all_row = [e[0] for e in all_edges]
        all_col = [e[1] for e in all_edges]
        self.all_inter_matrix = sp.coo_matrix((np.ones(len(all_row)), (all_row, all_col)),
                                              shape=[self.user_count + 1, self.item_count + 1])

    def behavior_dataset(self):
        return BehaviorDataset(self.user_count, self.item_count, self.pos_sampling, self.neg_count,
                            self.train_behavior_dict, self.behaviors)

    def validate_dataset(self):
        return TestDataset(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDataset(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))