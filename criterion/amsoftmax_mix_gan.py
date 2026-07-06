#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import accuracy
from helper.synth_table import slerp, PersistentSynthState


class amsoftmax_gan(nn.Module):
    """AM-Softmax + CAARMA joint-L_syn, with optional PERSISTENT synthetic classes.

    persistence=False  -> SLERP one-shot mixup (the matched ablation control):
        batch-local NN pairing, fresh per-batch synthetic classes whose prototype
        is the SLERP midpoint of the two real prototypes (recomputed, discarded).

    persistence=True   -> persistent synthetic classes:
        a stable global-NN pair->column map (helper.synth_table) gives every
        synthetic identity a LEARNABLE persistent prototype W_syn[:, col] that
        accumulates across batches; each visit SLERP-interpolates DIFFERENT real
        utterance embeddings (anchor from the batch, partner from a memory bank),
        so the model learns the synthetic speaker's DISTRIBUTION instead of a
        single one-shot point. joint-L_syn classifies synthetic samples against
        [ real prototypes ; activated persistent synthetic prototypes ].

        pair_strategy="fixed_nn" is the v1 control. pair_strategy="crp" is v2:
        each anchor speaker chooses between reusing an introduced top-k pair and
        creating a new top-k pair, with crp_alpha controlling the new-pair mass.
    """

    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30,
                 persistence=False, slerp_t=0.5, synth_bank_size=10,
                 synth_max_factor=4, pair_strategy="fixed_nn", crp_alpha=1.0,
                 crp_topk=4, **kwargs):
        super(amsoftmax_gan, self).__init__()
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.num_real = int(num_classes)
        self.ce = nn.CrossEntropyLoss()

        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

        self.persistence = bool(persistence)
        self.slerp_t = float(slerp_t)
        self._cached_synth = None
        self._cached_cols = None
        self.last_synth_cols = []
        self.last_synth_conditions = None
        if self.persistence:
            self.max_cols = int(synth_max_factor) * self.num_real
            # persistent learnable synthetic-class prototypes (DDP-synced).
            self.W_syn = torch.nn.Parameter(torch.randn(embedding_dim, self.max_cols),
                                            requires_grad=True)
            nn.init.xavier_normal_(self.W_syn, gain=1)
            # cross-batch state (pairing + per-speaker memory bank); plain python.
            self.synth = PersistentSynthState(self.num_real, self.max_cols,
                                              bank_size=synth_bank_size,
                                              pair_strategy=pair_strategy,
                                              crp_alpha=crp_alpha,
                                              crp_topk=crp_topk)
            print('Initialised PERSISTENT AM-Softmax m=%.3f s=%.3f slerp_t=%.2f '
                  'max_cols=%d bank=%d pair_strategy=%s crp_alpha=%.3f crp_topk=%d'
                  % (self.m, self.s, self.slerp_t, self.max_cols,
                     synth_bank_size, pair_strategy, crp_alpha, crp_topk))
        else:
            print('Initialised AM-Softmax (one-shot SLERP) m=%.3f s=%.3f slerp_t=%.2f'
                  % (self.m, self.s, self.slerp_t))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def get_extra_state(self):
        """Persist non-Parameter synthetic-class state in checkpoints."""
        if not self.persistence:
            return {"persistence": False}
        return {
            "persistence": True,
            "synth": self.synth.state_dict(),
        }

    def set_extra_state(self, state):
        """Restore synthetic-class state when loading a checkpoint."""
        if not state or not getattr(self, "persistence", False):
            return
        synth_state = state.get("synth")
        if synth_state is not None:
            self.synth.load_state_dict(synth_state)
        self._cached_synth = None
        self._cached_cols = None
        self.last_synth_cols = []
        self.last_synth_conditions = None

    # ------------------------------------------------------------------ utils
    def _am_loss(self, x_emb, W_cols, target):
        """AM-Softmax CE + top-1 acc. x_emb:(N,D) W_cols:(D,K) target:(N,) into K."""
        x_norm = F.normalize(x_emb, dim=1)
        w_norm = F.normalize(W_cols, dim=0)
        costh = torch.mm(x_norm, w_norm)                      # (N, K)
        delt = torch.zeros_like(costh).scatter_(1, target.view(-1, 1), self.m)
        costh_m_s = self.s * (costh - delt)
        loss = self.ce(costh_m_s, target)
        acc = accuracy(costh_m_s.detach(), target.detach(), topk=(1,))[0]
        return loss, acc

    # -------------------------------------------------------- one-shot (OFF)
    def _gen_oneshot_slerp(self, x, label):
        """Batch-local NN pairing, fresh per-batch synthetic classes, SLERP."""
        device = x.device
        labels = [int(l) for l in label.tolist()]
        set_label = list(set(labels))
        idx_of = {}
        for bi, l in enumerate(labels):
            idx_of.setdefault(l, []).append(bi)

        dic_spk = {}
        for s in set_label:
            cand = [k for k in set_label if k != s]
            if not cand:
                dic_spk[s] = s
                continue
            d = torch.stack([torch.dist(self.W[:, s], self.W[:, k]) for k in cand])
            dic_spk[s] = cand[int(torch.argmin(d))]

        B = x.size(0)
        w_mix = torch.zeros(self.W.size(0), B, device=device)
        y_mix = torch.zeros(B, dtype=torch.int64, device=device)
        samples, newlabel, labelid = [], {}, 0
        for bi in range(B):
            l1, l2 = labels[bi], dic_spk[labels[bi]]
            key = (min(l1, l2), max(l1, l2))
            proto = slerp(self.W[:, l1], self.W[:, l2], self.slerp_t)
            if key not in newlabel:
                newlabel[key] = labelid
                w_mix[:, labelid] = proto
                labelid += 1
            else:
                w_mix[:, newlabel[key]] = proto
            y_mix[bi] = newlabel[key]
            samples.append(slerp(x[bi], x[idx_of[l2][0]], self.slerp_t))
        synthetic = torch.stack(samples, 0)
        return synthetic, y_mix, w_mix[:, :labelid]

    # -------------------------------------------------------- persistent (ON)
    def _gen_persistent(self, x, label, update_state):
        """SLERP of anchor and partner under fixed-NN or CRP pair selection.

        Returns synthetic embeddings and persistent W_syn columns. In CRP mode,
        visits are committed only when a usable partner embedding exists.
        """
        state = self.synth
        labels = [int(l) for l in label.tolist()]
        idx_in_batch = {}
        for bi, l in enumerate(labels):
            idx_in_batch.setdefault(l, []).append(bi)

        if update_state:
            state.begin_batch_stats(x.size(0))

        samples, cols = [], []
        for bi, s in enumerate(labels):
            key, col, j, event = state.select_pair(s)
            if col is None:
                continue
            e_j, source = state.partner_embedding(j, idx_in_batch, x)
            if e_j is None:
                continue
            samples.append(slerp(x[bi], e_j, self.slerp_t))
            cols.append(int(col))
            if update_state:
                state.commit_visit(key, col, event, source)

        if len(samples) == 0:                       # degenerate-batch guard
            B = x.size(0)
            key, col0, _, event = state.select_pair(labels[0])
            if key is None or col0 is None:
                raise RuntimeError(
                    "Persistent synthetic generation failed: no assigned "
                    "synthetic column is available. Increase synth_max_factor "
                    "or reduce crp_topk."
                )
            partner = x[1] if B > 1 else x[0]
            samples.append(slerp(x[0], partner, self.slerp_t))
            cols.append(int(col0))
            if update_state:
                source = "batch" if B > 1 else "self"
                state.commit_visit(key, col0, event, source)

        if update_state:
            state.update_bank(x.detach(), label)
            state.finalize_batch_stats()
        return torch.stack(samples, 0), cols

    # ------------------------------------------------------------- forward
    def forward(self, x, label=None, flagSyn=False, update_state=False):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        if self.persistence:
            if flagSyn and self._cached_synth is not None:
                synthetic, cols = self._cached_synth, self._cached_cols
            else:
                synthetic, cols = self._gen_persistent(x, label, update_state)
                if update_state:
                    # Reuse the exact same stochastic CRP samples for L_syn and
                    # adversarial-G in this training step.
                    self._cached_synth = synthetic
                    self._cached_cols = cols
            self.last_synth_cols = [int(c) for c in cols]
            self.last_synth_conditions = None
        else:
            synthetic, y_oneshot, w_oneshot = self._gen_oneshot_slerp(x, label)
            self.last_synth_cols = []
            self.last_synth_conditions = w_oneshot[:, y_oneshot].detach().t()

        if not flagSyn:
            # L_real over the real prototypes only (avoid double-counting; L_syn
            # handles the synthetic-vs-real contrast).
            loss, acc = self._am_loss(x, self.W, label)
            return loss, acc, synthetic

        # ---- joint-L_syn: synthetic samples vs [ real ; synthetic ] ----------
        if self.persistence:
            active = sorted(self.synth.activated_cols)
            if len(active) == 0:
                # nothing activated yet: keep W_syn in the graph (DDP-safe) ~0 loss
                loss = 0.0 * self.W_syn.sum() + 0.0 * self.W.sum()
                self._cached_synth = None
                self._cached_cols = None
                return loss, torch.tensor(0.0, device=x.device), synthetic
            pos = {c: i for i, c in enumerate(active)}
            W_cols = torch.cat([self.W, self.W_syn[:, active]], dim=1)
            target = torch.tensor([self.num_real + pos[c] for c in cols],
                                  device=x.device, dtype=torch.int64)
            loss, acc = self._am_loss(synthetic, W_cols, target)
            # touch all of W_syn so every rank marks it 'used' every step (DDP).
            loss = loss + 0.0 * self.W_syn.sum()
            self._cached_synth = None
            self._cached_cols = None
            return loss, acc, synthetic
        else:
            W_cols = torch.cat([self.W, w_oneshot], dim=1)
            target = (y_oneshot + self.num_real)
            loss, acc = self._am_loss(synthetic, W_cols, target)
            return loss, acc, synthetic
