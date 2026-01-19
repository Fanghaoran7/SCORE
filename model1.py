#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import DataSet
from utils import BPRLoss, EmbLoss
from lightgcn import LightGCN

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class FiLMGating(nn.Module):
    def __init__(self, embed_dim, bhv_dim, dropout=0.1):
        super(FiLMGating, self).__init__()
        self.gamma_gen = nn.Sequential(
            nn.Linear(embed_dim * 2 + bhv_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.beta_gen = nn.Sequential(
            nn.Linear(embed_dim * 2 + bhv_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gamma_norm = nn.LayerNorm(embed_dim)

    def forward(self, u_emb, i_emb, bhv_emb):
        x = torch.cat([u_emb, i_emb, bhv_emb], dim=-1)
        gamma = self.gamma_norm(self.gamma_gen(x))
        beta = self.beta_gen(x)
        u_mod = gamma * u_emb + beta
        return u_mod


class DistortionAwareFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super(DistortionAwareFusion, self).__init__()
        self.embed_dim = embed_dim
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, u_gen, u_point):
        query = u_gen.unsqueeze(1)
        key = u_point.unsqueeze(1)
        value = u_point.unsqueeze(1)
        attn_output, _ = self.cross_attention(query, key, value)
        fused_emb = self.layer_norm1(query + attn_output)
        ffn_output = self.ffn(fused_emb)
        final_user_emb = self.layer_norm2(fused_emb + ffn_output)
        return final_user_emb.squeeze(1)


class SCORE(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(SCORE, self).__init__()
        self.device = args.device
        self.layers = args.layers
        self.embedding_size = args.embedding_size
        self.reg_weight = args.reg_weight
        self.log_reg = args.log_reg
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.gcn_batch_size = getattr(args, 'gcn_batch_size', 4096)
        self.embed_batch_size = getattr(args, 'embed_batch_size', 2048)
        self.freeze_gcn_after = getattr(args, 'freeze_gcn_after', -1)
        self.current_epoch = 0

        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix
        self.user_item_inter_set = dataset.user_item_inter_set
        self.test_users = list(dataset.test_interacts.keys())
        self.behaviors = args.behaviors

        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.bhv_embs = nn.Parameter(torch.eye(len(self.behaviors)))

        self.global_Graph = LightGCN(self.device, self.layers, self.n_users + 1, 
                                     self.n_items + 1, dataset.all_inter_matrix)
        self.behavior_Graph = LightGCN(self.device, self.layers, self.n_users + 1, 
                                       self.n_items + 1, dataset.inter_matrix[-1])

        self.film_gate = FiLMGating(
            embed_dim=self.embedding_size,
            bhv_dim=len(self.behaviors),
            dropout=args.message_dropout
        )

        self.fusion_module = DistortionAwareFusion(
            embed_dim=self.embedding_size,
            num_heads=args.num_heads,
            dropout=args.message_dropout
        )

        self.purchase_weight = getattr(args, 'purchase_weight', 0.3)
        purchase_input_dim = self.embedding_size * 2 + len(self.behaviors)
        self.purchase_predictor = nn.Sequential(
            nn.Linear(purchase_input_dim, self.embedding_size * 2),
            nn.ReLU(),
            nn.Dropout(args.message_dropout),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )

        self._load_user_purchase_info(dataset)

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.cross_loss = nn.BCELoss()
        self.purchase_loss = nn.BCELoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)
        self._load_model()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def _load_user_purchase_info(self, dataset):
        if hasattr(dataset, 'train_behavior_dict'):
            buy_dict = dataset.train_behavior_dict.get(self.behaviors[-1], {})
        else:
            buy_dict = {}
        
        self.user_purchase_labels = torch.zeros(self.n_users + 1, device=self.device)
        for user_str, items in buy_dict.items():
            user_id = int(user_str)
            if items:
                self.user_purchase_labels[user_id] = 1.0

    def compute_purchase_tendency(self, user_embs, item_embs, behavior_embs):
        if user_embs.dim() == 3:
            user_embs = user_embs.squeeze(1)
        if item_embs.dim() == 3:
            item_embs = item_embs.squeeze(1)
        if behavior_embs.dim() == 3:
            behavior_embs = behavior_embs.squeeze(1)
        
        combined = torch.cat([user_embs, item_embs, behavior_embs], dim=-1)
        purchase_prob = self.purchase_predictor(combined)
        return purchase_prob.squeeze()

    def _freeze_gcn_layers(self):
        if self.freeze_gcn_after > 0 and self.current_epoch >= self.freeze_gcn_after:
            for param in self.global_Graph.parameters():
                param.requires_grad = False
            for param in self.behavior_Graph.parameters():
                param.requires_grad = False

    def user_agg_item(self, user_samples, u_emb, ini_item_embs):
        keys = user_samples.tolist()
        user_item_set = self.user_item_inter_set[-1]
        agg_items = [user_item_set[x] for x in keys]

        max_len = max(len(l) for l in agg_items if l) or 1
        padded_list = np.zeros((len(agg_items), max_len), dtype=int)
        for i, l in enumerate(agg_items):
            if l:
                padded_list[i, :len(l)] = l
        padded_list = torch.from_numpy(padded_list).to(self.device)

        mask = (padded_list == 0)
        agg_item_emb = ini_item_embs[padded_list.long()]
        u_in = u_emb.repeat(1, max_len, 1)
        bhv_emb = self.bhv_embs[-1].repeat(u_in.shape[0], u_in.shape[1], 1)

        u_final = self.film_gate(u_in, agg_item_emb, bhv_emb)
        u_final[mask] = 0
        u_final = torch.sum(u_final, dim=1)
        return u_final, None

    def forward(self, batch_data):
        self._freeze_gcn_layers()
        
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.global_Graph(all_embeddings)
        user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
        buy_embeddings = self.behavior_Graph(all_embeddings)
        user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings, [self.n_users + 1, self.n_items + 1])

        p_samples = batch_data[:, 0, :]
        n_samples = batch_data[:, 1:-1, :].reshape(-1, 4)
        samples = torch.cat([p_samples, n_samples], dim=0)
        u_sample, i_samples, b_samples, gt_samples = torch.chunk(samples, 4, dim=-1)
        u_emb_log = user_embedding[u_sample.long()].squeeze()
        i_emb_log = item_embedding[i_samples.squeeze().long()]
        bhv_emb = self.bhv_embs[b_samples.reshape(-1).long()]
        u_final_log = self.film_gate(u_emb_log, i_emb_log, bhv_emb)
        log_loss_scores = torch.sum((u_final_log * i_emb_log), dim=-1).unsqueeze(1)
        log_loss = self.cross_loss(torch.sigmoid(log_loss_scores), gt_samples.float())

        pair_samples = batch_data[:, -1, :-1]
        mask = torch.any(pair_samples != 0, dim=-1)
        pair_samples = pair_samples[mask]
        bpr_loss = 0
        purchase_tendency_loss = torch.tensor(0.0, device=self.device)
        
        if pair_samples.shape[0] > 0:
            user_samples = pair_samples[:, 0].long()
            item_samples = pair_samples[:, 1:].long()
            u_emb = user_embedding[user_samples]
            i_emb = item_embedding[item_samples]

            u_point, _ = self.user_agg_item(user_samples, u_emb.unsqueeze(1), item_embedding)
            u_point = u_point.squeeze(1)
            u_gen = u_emb + user_buy_embedding[user_samples]
            i_final = i_emb + item_buy_embedding[item_samples]
            final_user_emb = self.fusion_module(u_gen, u_point)

            scores = torch.sum(final_user_emb.unsqueeze(1) * i_final, dim=-1)
            p_scores, n_scores = torch.chunk(scores, 2, dim=-1)
            bpr_loss += self.bpr_loss(p_scores.squeeze(), n_scores.squeeze())

            purchase_users = user_samples
            purchase_items = item_samples[:, 0]
            
            user_embs = user_embedding[purchase_users]
            item_embs = item_embedding[purchase_items]
            behavior_embs = self.bhv_embs[-1].unsqueeze(0).repeat(len(purchase_users), 1)

            purchase_probs = self.compute_purchase_tendency(user_embs, item_embs, behavior_embs)
            user_has_purchased = self.user_purchase_labels[purchase_users]
            purchase_tendency_loss = self.purchase_loss(purchase_probs, user_has_purchased)

        emb_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        base_loss = self.log_reg * log_loss + (1 - self.log_reg) * bpr_loss + self.reg_weight * emb_loss
        total_loss = base_loss + self.purchase_weight * purchase_tendency_loss
        
        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None or self.storage_item_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.global_Graph(all_embeddings)
            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            buy_embeddings = self.behavior_Graph(all_embeddings)
            user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings, [self.n_users + 1, self.n_items + 1])

            storage_u_point = torch.zeros(self.n_users + 1, self.embedding_size)
            test_users = [int(x) for x in self.test_users]
            tmp_emb_list = []
            test_batch_size = self.embed_batch_size
            
            for i in range(0, len(test_users), test_batch_size):
                tmp_users = test_users[i: i + test_batch_size]
                tmp_users_tensor = torch.LongTensor(tmp_users).to(self.device)
                tmp_u_emb = user_embedding[tmp_users_tensor].unsqueeze(1)
                tmp_u_point, _ = self.user_agg_item(tmp_users_tensor, tmp_u_emb, item_embedding)
                tmp_emb_list.append(tmp_u_point.squeeze(1).cpu())
                torch.cuda.empty_cache()
            
            storage_u_point[test_users] = torch.cat(tmp_emb_list, dim=0)

            storage_u_gen = (user_embedding + user_buy_embedding).cpu()

            test_users_tensor = torch.LongTensor(test_users)
            u_gen_test = storage_u_gen[test_users_tensor].to(self.device)
            u_point_test = storage_u_point[test_users_tensor].to(self.device)
            final_test_user_embs = self.fusion_module(u_gen_test, u_point_test)

            self.storage_user_embeddings = torch.zeros(self.n_users + 1, self.embedding_size)
            self.storage_user_embeddings[test_users_tensor] = final_test_user_embs.cpu()
            self.storage_item_embeddings = (item_embedding + item_buy_embedding).cpu()

        user_emb = self.storage_user_embeddings[users.cpu().long()].to(self.device)
        item_emb = self.storage_item_embeddings.to(self.device)
        scores = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return scores