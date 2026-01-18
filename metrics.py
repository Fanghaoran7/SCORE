#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import mean_absolute_error, mean_squared_error


def hit_(pos_index, pos_len):
    result = np.cumsum(pos_index, axis=1)
    return (result > 0).astype(int)


def mrr_(pos_index, pos_len):
    idxs = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=float)
    for row, idx in enumerate(idxs):
        if pos_index[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, idx:] = 0
    return result


def map_(pos_index, pos_len):
    pre = precision_(pos_index, pos_len)
    sum_pre = np.cumsum(pre * pos_index.astype(float), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=float)
    for row, lens in enumerate(actual_len):
        ranges = np.arange(1, pos_index.shape[1] + 1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result


def recall_(pos_index, pos_len):
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg_(pos_index, pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result


def precision_(pos_index, pos_len):
    return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)


def gauc_(user_len_list, pos_len_list, pos_rank_sum):
    neg_len_list = user_len_list - pos_len_list
    
    any_without_pos = np.any(pos_len_list == 0)
    any_without_neg = np.any(neg_len_list == 0)
    non_zero_idx = np.full(len(user_len_list), True, dtype=bool)
    
    if any_without_pos:
        non_zero_idx *= (pos_len_list != 0)
    if any_without_neg:
        non_zero_idx *= (neg_len_list != 0)
    if any_without_pos or any_without_neg:
        item_list = user_len_list, neg_len_list, pos_len_list, pos_rank_sum
        user_len_list, neg_len_list, pos_len_list, pos_rank_sum = \
            map(lambda x: x[non_zero_idx], item_list)

    pair_num = (user_len_list + 1) * pos_len_list - pos_len_list * (pos_len_list + 1) / 2 - np.squeeze(pos_rank_sum)
    user_auc = pair_num / (neg_len_list * pos_len_list)
    result = (user_auc * pos_len_list).sum() / pos_len_list.sum()
    return result


def auc_(trues, preds):
    fps, tps = _binary_clf_curve(trues, preds)

    if len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return sk_auc(fpr, tpr)


def mae_(trues, preds):
    return mean_absolute_error(trues, preds)


def rmse_(trues, preds):
    return np.sqrt(mean_squared_error(trues, preds))


def log_loss_(trues, preds):
    eps = 1e-15
    preds = np.float64(preds)
    preds = np.clip(preds, eps, 1 - eps)
    loss = np.sum(-trues * np.log(preds) - (1 - trues) * np.log(1 - preds))
    return loss / len(preds)


def _binary_clf_curve(trues, preds):
    trues = (trues == 1)
    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps


metrics_dict = {
    'ndcg': ndcg_,
    'hit': hit_,
    'precision': precision_,
    'map': map_,
    'recall': recall_,
    'mrr': mrr_,
    'rmse': rmse_,
    'mae': mae_,
    'logloss': log_loss_,
    'auc': auc_,
    'gauc': gauc_
}