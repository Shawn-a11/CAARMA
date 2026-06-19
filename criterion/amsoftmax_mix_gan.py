#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy
from helper.mixup_avg import mixup_data_euc_avg

class amsoftmax_gan(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30, **kwargs):
        super(amsoftmax_gan, self).__init__()

        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        size = self.W.shape[1]  # The size of the diagonal matrix
        self.I = torch.diag(torch.ones(size))
        if torch.cuda.is_available():
            self.I = self.I.to('cuda:0')

        print('Initialised AM-Softmax m=%.3f s=%.3f'%(self.m, self.s))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def forward(self, x, label=None, flagSyn=False):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
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