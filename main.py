#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
import gc

from dataset import DataSet
from model import SCORE
from trainer import Trainer

torch.cuda.empty_cache()
gc.collect()

SEED = 2021
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(SEED)


def run_single_trial(args: argparse.Namespace):
    args.disable_tqdm = False
    args.TIME = time.strftime("%Y-%m-%d_%H_%M_%S")

    os.makedirs(os.path.join('./log', args.model_name), exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    logfile = f'{args.data_name}_emb_{args.embedding_size}_{args.TIME}'
    logger.add(f'./log/{args.model_name}/{logfile}.log', encoding='utf-8')

    logger.info("="*60)
    logger.info("Experiment Settings")
    logger.info("="*60)
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("="*60)

    if torch.cuda.is_available():
        try:
            gpu_idx = int(args.device.split(':')[-1])
        except:
            gpu_idx = 0
        try:
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        except Exception as e:
            logger.warning(f"Could not get GPU properties: {e}")

    torch.cuda.empty_cache()
    gc.collect()

    dataset = DataSet(args)
    model = SCORE(args, dataset).to(args.device)
    
    if args.gpu_no > 1:
        model = nn.DataParallel(model, device_ids=[i for i in range(args.gpu_no)])

    trainer = Trainer(model, dataset, args)
    logger.info(model)
    logger.info(f"Number of trainable parameters: "
               f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    trainer.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SCORE Model', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--gcn_batch_size', type=int, default=4096)
    parser.add_argument('--embed_batch_size', type=int, default=2048)
    parser.add_argument('--freeze_gcn_after', type=int, default=-1)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--pin_memory', type=bool, default=True)

    parser.add_argument('--purchase_weight', type=float, default=0.3)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--reg_weight', type=float, default=1e-4)
    parser.add_argument('--log_reg', type=float, default=0.5)
    parser.add_argument('--node_dropout', type=float, default=0.4)
    parser.add_argument('--message_dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--data_name', type=str, default='tmall',
                       choices=['tmall', 'yelp', 'taobao'])
    parser.add_argument('--neg_count', type=int, default=8)
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80])
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'])
    parser.add_argument('--test_batch_size', type=int, default=1024)

    parser.add_argument('--gpu_no', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='./check_point')
    parser.add_argument('--model_name', type=str, default='SCORE')
    parser.add_argument('--if_load_model', type=bool, default=False)
    parser.add_argument('--check_point', type=str, default='')

    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--metric_decimals', type=int, default=6)

    args = parser.parse_args()

    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'yelp':
        args.data_path = './data/Yelp'
        args.behaviors = ['tip', 'neutral', 'neg', 'pos']
    elif args.data_name == 'taobao':
        args.data_path = './data/taobao'
        args.behaviors = ['view', 'cart', 'buy']
    else:
        raise ValueError(f'Unknown dataset: {args.data_name}')

    if torch.cuda.is_available():
        try:
            gpu_idx = int(args.device.split(':')[-1])
        except:
            gpu_idx = 0
        try:
            total_memory = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9
            
            if total_memory < 8:
                args.embedding_size = min(args.embedding_size, 64)
                args.layers = min(args.layers, 1)
                args.batch_size = min(args.batch_size, 512)
                args.test_batch_size = min(args.test_batch_size, 512)
                args.gradient_accumulation_steps = max(args.gradient_accumulation_steps, 4)
                args.gcn_batch_size = min(args.gcn_batch_size, 2048)
            elif total_memory < 16:
                args.embedding_size = min(args.embedding_size, 128)
                args.layers = min(args.layers, 4)
                args.batch_size = min(args.batch_size, 1024)
                args.gradient_accumulation_steps = max(args.gradient_accumulation_steps, 2)
        except Exception as e:
            pass

    logger.info("="*60)
    logger.info("Starting Single Training Trial")
    logger.info("="*60)
    run_single_trial(args)