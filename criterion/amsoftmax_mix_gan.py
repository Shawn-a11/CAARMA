#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy
from helper.mixup_avg import mixup_data_euc_avg


class amsoftmax_gan(nn.Module):
    """Sub-center AM-Softmax (+ CAARMA joint-L_syn).

    Each speaker has K sub-centres; the class score is the MAX cosine over its
    K sub-centres, so a sample only needs to match its nearest sub-centre. A
    speaker's real intra-class variation (sessions / conditions) can occupy
    different sub-centres instead of being squashed to one point — the deep
    metric-learning literature links preserved intra-class diversity to better
    generalisation to unseen classes (zero-shot). K=1 recovers vanilla AM-Softmax.

    CAARMA's mixup needs one prototype per speaker, so it is fed the per-speaker
    MEAN over sub-centres; classification (L_real and joint-L_syn) uses the
    max-pool over sub-centres.
    """

    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30,
                 num_subcenters=3, **kwargs):
        super(amsoftmax_gan, self).__init__()
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.num_classes = num_classes
        self.K = int(num_subcenters)
        # (D, C, K) sub-centre prototypes; random init breaks K-symmetry.
        self.W = torch.nn.Parameter(
            torch.randn(embedding_dim, num_classes, self.K), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.I = torch.diag(torch.ones(num_classes))           # kept (unused, legacy)
        if torch.cuda.is_available():
            self.I = self.I.to('cuda:0')
        print('Initialised sub-center AM-Softmax m=%.3f s=%.3f K=%d'
              % (self.m, self.s, self.K))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def _proto(self):
        # per-speaker single prototype for mixup = mean over sub-centres -> (D, C)
        return self.W.mean(dim=2)

    def _subcenter_cos(self, e_norm):
        # e_norm: (N, D) unit rows. self.W: (D, C, K). -> max-pooled cos (N, C).
        Wn = F.normalize(self.W, dim=0)                    # each sub-centre unit
        cos = torch.einsum('nd,dck->nck', e_norm, Wn)      # (N, C, K)
        return cos.max(dim=2).values                       # (N, C)

    def _am_loss(self, costh, target):
        # additive margin on the true column + CE. costh:(N,cols) target:(N,)
        lv = target.view(-1, 1)
        if lv.is_cuda:
            lv = lv.cpu()
        delt = torch.zeros(costh.size()).scatter_(1, lv, self.m).to(costh.device)
        costh_m_s = self.s * (costh - delt)
        loss = self.ce(costh_m_s, target)
        acc = accuracy(costh_m_s.detach(), target.detach(), topk=(1,))[0]
        return loss, acc

    def forward(self, x, label=None, flagSyn=False):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        # mixup uses the per-speaker MEAN sub-centre prototype.
        synthetic_embeddings, y_combined, w_combined = mixup_data_euc_avg(
            x, self._proto(), label)

        if flagSyn:
            # joint-L_syn with sub-centres: classify synthetic embeddings against
            # [ real classes (sub-centre max-pool) ; synthetic classes (1 proto) ].
            # Real prototypes are negatives -> push nearest real speakers apart.
            e_norm = F.normalize(synthetic_embeddings.to(x.device), dim=1)   # (Ns, D)
            cos_real = self._subcenter_cos(e_norm)                          # (Ns, C)
            w_syn = F.normalize(w_combined.to(x.device), dim=0)             # (D, labelid)
            cos_syn = torch.mm(e_norm, w_syn)                              # (Ns, labelid)
            costh = torch.cat([cos_real, cos_syn], dim=1)                  # (Ns, C+labelid)
            target = (y_combined + self.num_classes).to(x.device)          # synthetic cols
            loss, acc = self._am_loss(costh, target)
            return loss, acc, synthetic_embeddings
        else:
            e_norm = F.normalize(x, dim=1)                                 # (N, D)
            costh = self._subcenter_cos(e_norm)                           # (N, C)
            loss, acc = self._am_loss(costh, label)
            return loss, acc, synthetic_embeddings
