#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy
from helper.mixup_avg import mixup_data_euc_avg

class amsoftmax_gan(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        margin=0.2,
        scale=30,
        prototype_only_virtual=False,
        virtual_negative_topk=4,
        virtual_negative_t=0.5,
        virtual_negatives_per_batch=8,
        **kwargs,
    ):
        super(amsoftmax_gan, self).__init__()

        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.prototype_only_virtual = bool(prototype_only_virtual)
        self.virtual_negative_topk = int(virtual_negative_topk)
        self.virtual_negative_t = float(virtual_negative_t)
        self.virtual_negatives_per_batch = int(virtual_negatives_per_batch)
        self.last_virtual_count = 0

        if not 0.0 <= self.virtual_negative_t <= 1.0:
            raise ValueError('virtual_negative_t must be in [0, 1].')
        if self.virtual_negative_topk < 1:
            raise ValueError('virtual_negative_topk must be at least 1.')
        if self.virtual_negatives_per_batch < 0:
            raise ValueError('virtual_negatives_per_batch must be non-negative.')

        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM-Softmax m=%.3f s=%.3f'%(self.m, self.s))
        if self.prototype_only_virtual:
            print(
                'Prototype-only virtual negatives: '
                'top-k=%d t=%.2f per_batch=%d (no synthetic positives)' % (
                    self.virtual_negative_topk,
                    self.virtual_negative_t,
                    self.virtual_negatives_per_batch,
                )
            )
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    @staticmethod
    def _slerp_rows(start, end, t):
        """Spherical interpolation for two batches of unit row vectors."""
        dot = (start * end).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        safe_denom = sin_omega.clamp_min(1e-7)
        spherical = (
            torch.sin((1.0 - t) * omega) / safe_denom * start
            + torch.sin(t * omega) / safe_denom * end
        )
        linear = (1.0 - t) * start + t * end
        mixed = torch.where(sin_omega.abs() < 1e-4, linear, spherical)
        return F.normalize(mixed, p=2, dim=1)

    def _build_virtual_negative_weights(self, label):
        """Create detached SLERP prototypes that have no positive examples."""
        num_classes = self.W.size(1)
        if num_classes < 2 or self.virtual_negatives_per_batch == 0:
            return self.W.new_empty((self.in_feats, 0))

        with torch.no_grad():
            anchors = torch.unique(label.detach(), sorted=False)
            if anchors.numel() > self.virtual_negatives_per_batch:
                order = torch.randperm(anchors.numel(), device=anchors.device)
                anchors = anchors[order[:self.virtual_negatives_per_batch]]

            weights = F.normalize(self.W.detach(), p=2, dim=0)
            similarity = weights[:, anchors].T @ weights
            similarity.scatter_(1, anchors.view(-1, 1), float('-inf'))

            k = min(self.virtual_negative_topk, num_classes - 1)
            neighbours = similarity.topk(k=k, dim=1, largest=True).indices
            choices = torch.randint(k, (anchors.numel(),), device=anchors.device)
            partners = neighbours[
                torch.arange(anchors.numel(), device=anchors.device), choices
            ]

            virtual = self._slerp_rows(
                weights[:, anchors].T,
                weights[:, partners].T,
                self.virtual_negative_t,
            )
            return virtual.T.contiguous()

    def _real_amsoftmax_loss(self, x, label, virtual_weights=None):
        """Classify real speech; optional virtual columns are denominator-only."""
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.W, p=2, dim=0)
        real_costh = torch.mm(x_norm, w_norm)
        margin = torch.zeros_like(real_costh)
        margin.scatter_(1, label.view(-1, 1), self.m)
        logits = real_costh - margin

        if virtual_weights is not None and virtual_weights.numel() > 0:
            virtual_weights = F.normalize(virtual_weights.detach(), p=2, dim=0)
            virtual_costh = torch.mm(x_norm, virtual_weights)
            logits = torch.cat((logits, virtual_costh), dim=1)

        logits = self.s * logits
        loss = self.ce(logits, label)
        acc = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        return loss, acc

    def forward(self, x, label=None, flagSyn=False):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        if self.prototype_only_virtual:
            if flagSyn:
                raise RuntimeError(
                    'prototype_only_virtual has no synthetic-positive loss.'
                )
            virtual_weights = self._build_virtual_negative_weights(label)
            self.last_virtual_count = virtual_weights.size(1)
            loss, acc = self._real_amsoftmax_loss(x, label, virtual_weights)
            empty_synthetic = x.new_empty((0, self.in_feats))
            return loss, acc, empty_synthetic

        synthetic_embeddings,  y_combined , w_combined = mixup_data_euc_avg(
            x, self.W, label
            )
        if flagSyn:
            
            # joint-L_syn: put the REAL prototypes W into the synthetic-class
            # softmax denominator (as negatives), so each synthetic embedding
            # (~ midpoint of a speaker pair) must be separable from real
            # speakers too. This directly pushes the nearest real prototypes
            # apart (margin), instead of the original syn-vs-syn-only softmax
            # where synthetic classes never contrast against real speakers.
            # Synthetic labels are offset by num_real so they index the
            # synthetic columns, not the first real-speaker columns. Only the
            # synthetic embeddings are classified here (L_real covers the real
            # ones, so we avoid double-counting).
            num_real = self.W.shape[1]
            x_combined_0 = synthetic_embeddings.to(x.device)
            w_combined_0 = torch.cat((self.W.to(x.device), w_combined.to(x.device)), dim=1)
            y_combined_0 = (y_combined + num_real).to(x.device)

            x_norm = torch.norm(x_combined_0, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x_combined_0, x_norm)
            w_norm = torch.norm(w_combined_0, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(w_combined_0, w_norm)
            costh = torch.mm(x_norm, w_norm)
            label_view = y_combined_0.view(-1,1) #label.view(-1, 1)
            if label_view.is_cuda: label_view = label_view.cpu()
            delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
            if x.is_cuda: delt_costh = delt_costh.cuda()
            costh_m = costh - delt_costh
            costh_m_s = self.s * costh_m
            
            loss = self.ce(costh_m_s, y_combined_0) #label)
            
            final_loss = (loss)
            acc = accuracy(costh_m_s.detach(), y_combined_0.detach(), topk=(1,))[0]
            return final_loss, acc, synthetic_embeddings
        else: 

            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x, x_norm)
            w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.W, w_norm)
            costh = torch.mm(x_norm, w_norm)
            label_view =  label.view(-1, 1)
            if label_view.is_cuda: label_view = label_view.cpu()
            delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
            if x.is_cuda: delt_costh = delt_costh.cuda()
            costh_m = costh - delt_costh
            costh_m_s = self.s * costh_m
            
            loss = self.ce(costh_m_s,  label)
            
            final_loss = (loss)
            acc = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
            return final_loss, acc, synthetic_embeddings
