#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import torch
import numpy as np
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from dataset import DataSet
from metrics import metrics_dict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class Trainer:
    def __init__(self, model, dataset: DataSet, args):
        self.model = model
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.patience = args.patience
        self.device = args.device
        self.disable_tqdm = args.disable_tqdm

        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 2)
        self.mixed_precision = getattr(args, 'mixed_precision', False)
        self.pin_memory = getattr(args, 'pin_memory', True)
        self.metric_decimals = int(getattr(args, 'metric_decimals', 6))

        self._prepare_purchase_labels()
        self.optimizer = self.get_optimizer(self.model)

        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _round_metrics(self, metric_dict: dict, decimals: int = None) -> dict:
        if decimals is None:
            decimals = self.metric_decimals
        out = {}
        for k, v in metric_dict.items():
            if isinstance(v, (float, np.floating, int, np.integer)):
                out[k] = round(float(v), decimals)
            else:
                out[k] = v
        return out

    def _get_selection_key(self) -> str:
        k = self.topk[0]
        metrics_lower = [m.lower() for m in self.metrics]
        if 'ndcg' in metrics_lower:
            return f'ndcg@{k}'
        return f'hit@{k}'

    def _prepare_purchase_labels(self):
        self.purchase_behavior_idx = len(self.behaviors) - 1
        self.val_purchase_labels = self._load_purchase_labels('validation')
        self.test_purchase_labels = self._load_purchase_labels('test')

        if not self.disable_tqdm:
            val_positive = sum(self.val_purchase_labels.values())
            test_positive = sum(self.test_purchase_labels.values())
            logger.info(
                f"Purchase Labels - Validation: {val_positive}/{len(self.val_purchase_labels)} positive, "
                f"Test: {test_positive}/{len(self.test_purchase_labels)} positive"
            )

    def _load_purchase_labels(self, phase):
        purchase_labels = {}

        if phase == 'validation':
            behavior_dict = getattr(self.dataset, 'validation_behavior_dict', {})
            interact_dict = getattr(self.dataset, 'validation_interacts', {})
        else:
            behavior_dict = getattr(self.dataset, 'test_behavior_dict', {})
            interact_dict = getattr(self.dataset, 'test_interacts', {})

        buy_behavior = self.behaviors[-1]
        if buy_behavior in behavior_dict:
            purchase_dict = behavior_dict[buy_behavior]
        else:
            purchase_file = f"{phase}_buy_dict.txt"
            purchase_path = os.path.join(self.dataset.path, purchase_file)
            if os.path.exists(purchase_path):
                try:
                    with open(purchase_path, 'r', encoding='utf-8') as f:
                        purchase_dict = json.load(f)
                except Exception:
                    purchase_dict = {}
            else:
                purchase_dict = {}

        for user_id in interact_dict.keys():
            purchase_labels[user_id] = float(purchase_dict.get(user_id, 0.0))

        return purchase_labels

    def get_optimizer(self, model):
        params = model.module.parameters() if hasattr(model, 'module') else model.parameters()
        return optim.Adam(
            filter(lambda p: p.requires_grad, params),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def clear_parameter(self, model):
        target_model = model.module if hasattr(model, 'module') else model
        target_model.storage_user_embeddings = None
        target_model.storage_item_embeddings = None

    def train_model(self):
        train_loader = DataLoader(
            self.dataset.behavior_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=self.pin_memory
        )

        best_result = -float('inf')
        best_val_dict_raw = {}
        best_epoch = 0
        best_model = None
        final_test_raw = None

        selection_key = self._get_selection_key()
        if not self.disable_tqdm:
            logger.info(f"Model selection metric (raw): {selection_key}")

        for epoch in range(self.epochs):
            self.model.train()

            if hasattr(self.model, 'module'):
                self.model.module.current_epoch = epoch
            else:
                self.model.current_epoch = epoch

            test_metric_raw, validate_metric_raw = self._train_one_epoch(train_loader, epoch)

            current_metric = float(validate_metric_raw.get(selection_key, 0.0))

            if current_metric > best_result + 1e-12:
                best_result = current_metric
                best_val_dict_raw = validate_metric_raw
                best_model = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                final_test_raw = test_metric_raw
                if not self.disable_tqdm:
                    logger.info(
                        f"New best model @ epoch {epoch + 1}: {selection_key}={current_metric:.8f}"
                    )

            if epoch - best_epoch > self.patience:
                if not self.disable_tqdm:
                    logger.info(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch + 1}")
                break

        if best_model is not None:
            if not self.disable_tqdm:
                self.save_model(best_model)
                logger.info(f"Training finished. Best epoch: {best_epoch + 1}")
                
                val_metrics_display = {k: v for k, v in best_val_dict_raw.items() if not k.startswith('purchase_')}
                test_metrics_display = {k: v for k, v in final_test_raw.items() if not k.startswith('purchase_')}
                
                logger.info(f"Validation (rounded): {self._round_metrics(val_metrics_display, self.metric_decimals)}")
                if final_test_raw is not None:
                    logger.info(f"Test (rounded): {self._round_metrics(test_metrics_display, self.metric_decimals)}")

        return best_result

    def _train_one_epoch(self, data_loader, epoch):
        start_time = time.time()
        total_loss = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch_data in enumerate(
            tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"Train {epoch + 1}",
                disable=self.disable_tqdm
            )
        ):
            batch_data = batch_data.to(self.device)

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = self.model(batch_data)
                    if isinstance(loss, torch.Tensor):
                        loss = loss.mean() / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                loss = self.model(batch_data)
                if isinstance(loss, torch.Tensor):
                    loss = loss.mean() / self.gradient_accumulation_steps
                loss.backward()

            total_loss += float(loss.item()) * self.gradient_accumulation_steps

            if (batch_index + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        if (len(data_loader) % self.gradient_accumulation_steps) != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        total_loss /= max(len(data_loader), 1)
        epoch_time = time.time() - start_time
        if not self.disable_tqdm:
            logger.info(f'Epoch {epoch + 1} [{epoch_time:.2f}s]: Train loss: {total_loss:.6f}')

        self.clear_parameter(self.model)

        validate_metric_raw = self.evaluate_with_purchase(
            self.dataset.validate_dataset(),
            self.dataset.validation_interacts,
            self.dataset.validation_gt_length,
            phase='Validate'
        )
        test_metric_raw = self.evaluate_with_purchase(
            self.dataset.test_dataset(),
            self.dataset.test_interacts,
            self.dataset.test_gt_length,
            phase='Test'
        )
        return test_metric_raw, validate_metric_raw

    def evaluate_with_purchase(self, dataset, gt_interacts, gt_length, phase):
        rec_metrics_raw = self.evaluate_recommendation(dataset, gt_interacts, gt_length, phase)
        purchase_metrics = self.evaluate_purchase_tendency(dataset, phase)
        rec_metrics_raw.update(purchase_metrics)
        return rec_metrics_raw

    def evaluate_recommendation(self, dataset, gt_interacts, gt_length, phase):
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=self.pin_memory
        )
        self.model.eval()
        topk_list = []
        train_items = self.dataset.train_behavior_dict[self.behaviors[-1]]

        with torch.no_grad():
            for batch_users in data_loader:
                batch_users = batch_users.to(self.device)
                scores = self.model.full_predict(batch_users)

                for idx, user in enumerate(batch_users):
                    user_id_str = str(int(user.item()))
                    user_score = scores[idx].cpu()

                    if user_id_str in train_items:
                        user_score[train_items[user_id_str]] = -np.inf

                    _, topk_idx = torch.topk(user_score, max(self.topk))
                    gt_items = gt_interacts[user_id_str]
                    mask = np.isin(topk_idx.cpu().numpy(), gt_items)
                    topk_list.append(mask)

                torch.cuda.empty_cache()

        metric_raw = self.calculate_result(np.array(topk_list), gt_length)
        if not self.disable_tqdm:
            logger.info(f"{phase} Recommendation results (rounded): {self._round_metrics(metric_raw, self.metric_decimals)}")
        return metric_raw

    def evaluate_purchase_tendency(self, dataset, phase):
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=self.pin_memory
        )
        self.model.eval()

        all_preds, all_labels = [], []
        target_model = self.model.module if hasattr(self.model, 'module') else self.model

        with torch.no_grad():
            for batch_users in tqdm(
                data_loader,
                desc=f"Purchase Tendency {phase}",
                disable=self.disable_tqdm
            ):
                batch_users = batch_users.to(self.device)

                for user in batch_users:
                    user_id = int(user.item())
                    user_id_str = str(user_id)

                    try:
                        user_emb = target_model.user_embedding(user.unsqueeze(0))
                        item_id = int(np.random.randint(1, target_model.n_items))
                        item_emb = target_model.item_embedding(torch.tensor([item_id], device=self.device))
                        behavior_emb = target_model.bhv_embs[-1].unsqueeze(0)

                        purchase_prob = target_model.compute_purchase_tendency(
                            user_emb, item_emb, behavior_emb
                        )

                        if phase.lower() == 'validate':
                            label = float(self.val_purchase_labels.get(user_id_str, 0.0))
                        else:
                            label = float(self.test_purchase_labels.get(user_id_str, 0.0))

                        all_preds.append(float(purchase_prob.item()))
                        all_labels.append(label)

                    except Exception:
                        continue

                torch.cuda.empty_cache()

        purchase_metrics = self._calculate_purchase_metrics(all_preds, all_labels)
        return purchase_metrics

    def _calculate_purchase_metrics(self, preds, labels):
        if len(preds) == 0 or len(set(labels)) <= 1:
            return {
                'purchase_auc': 0.0,
                'purchase_f1': 0.0,
                'purchase_precision': 0.0,
                'purchase_recall': 0.0
            }

        preds_array = np.array(preds, dtype=float)
        labels_array = np.array(labels, dtype=float)

        try:
            auc = float(roc_auc_score(labels_array, preds_array))
        except Exception:
            auc = 0.0

        thresholds = np.arange(0.1, 1.0, 0.1)
        best_f1, best_precision, best_recall = 0.0, 0.0, 0.0

        for threshold in thresholds:
            binary_preds = (preds_array > threshold).astype(int)
            try:
                precision = float(precision_score(labels_array, binary_preds, zero_division=0))
                recall = float(recall_score(labels_array, binary_preds, zero_division=0))
                f1 = float(f1_score(labels_array, binary_preds, zero_division=0))
                if f1 > best_f1:
                    best_f1, best_precision, best_recall = f1, precision, recall
            except Exception:
                continue

        return {
            'purchase_auc': round(auc, self.metric_decimals),
            'purchase_f1': round(best_f1, self.metric_decimals),
            'purchase_precision': round(best_precision, self.metric_decimals),
            'purchase_recall': round(best_recall, self.metric_decimals)
        }

    def calculate_result(self, topk_list, gt_len):
        metric_dict = {}
        result_list = []

        for metric in self.metrics:
            metric_func = metrics_dict[metric.lower()]
            result_list.append(metric_func(topk_list, gt_len))

        result_list = np.stack(result_list, axis=0).mean(axis=1)

        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                metric_dict[f'{metric}@{topk}'] = float(value[topk - 1])

        return metric_dict

    def save_model(self, state_dict):
        model_name = f"{self.model_name}.pth"
        save_path = os.path.join(self.model_path, model_name)
        torch.save(state_dict, save_path)
        logger.info(f"Model saved to {save_path}")