"""Persistent synthetic-speaker machinery (professor's direction, 2026-06).

CAARMA generates each synthetic class `(i,j)` inside one batch and discards it
(one-shot) -> the model only ever sees a synthetic speaker as a single point and
cannot learn its distribution. This module makes synthetic classes PERSISTENT:

  * a stable global-NN pairing assigns every real speaker `s` a partner
    `j = NN(s)`, and every undirected pair `{s,j}` a STABLE column index that
    survives across batches/epochs;
  * a learnable prototype W_syn[:, col] (held by the criterion) accumulates
    training signal for that synthetic class across every visit;
  * each visit draws DIFFERENT real utterance embeddings of `s` and `j`
    (anchor from the current batch, partner from a per-speaker memory bank when
    absent) and SLERP-interpolates them -> "same synthetic class, different
    utterance samples".

The memory bank is the XBM / MoCo "slow-drift" mechanism (Wang et al. CVPR'20):
past embeddings approximate current ones, so stale partner embeddings are fine.
Persisting SYNTHETIC (interpolated) identities is the novelty vs MemVir
(2103.16940), which only persists REAL past classes.

CRP / stick-breaking create-vs-reuse control is deliberately NOT here yet: v1
tests the persistence core (matched persistence ON vs OFF); CRP is the next axis.
"""

from collections import defaultdict, deque

import torch
import torch.nn.functional as F


def slerp(p0, p1, t=0.5, eps=1e-7):
    """Spherical linear interpolation on the unit hypersphere.

    p0, p1: (..., D). Returns a unit vector (..., D). Falls back to normalised
    lerp when the two points are nearly (anti-)parallel (sin(omega) -> 0).
    """
    p0n = F.normalize(p0, dim=-1, eps=eps)
    p1n = F.normalize(p1, dim=-1, eps=eps)
    dot = (p0n * p1n).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    near = so.abs() < 1e-4
    a = torch.sin((1.0 - t) * omega) / so
    b = torch.sin(t * omega) / so
    out = a * p0n + b * p1n
    if near.any():                       # degenerate: plain normalised lerp
        lerp = F.normalize((1.0 - t) * p0n + t * p1n, dim=-1, eps=eps)
        out = torch.where(near, lerp, out)
    return F.normalize(out, dim=-1, eps=eps)


class PersistentSynthState:
    """Cross-batch state for persistent synthetic classes (not an nn.Module).

    Holds the stable pair->column map and the per-speaker embedding memory bank.
    The learnable prototypes W_syn live in the criterion (so DDP syncs them).
    All state here is either deterministic from the (DDP-synced) real prototypes
    W (the pairing) or a local approximation (the bank) -> rank-safe.
    """

    def __init__(self, num_real, max_cols, bank_size=10):
        self.num_real = int(num_real)
        self.max_cols = int(max_cols)
        self.bank_size = int(bank_size)
        self.bank = defaultdict(lambda: deque(maxlen=self.bank_size))  # spk -> recent detached emb (D,)
        self.pair_col = {}            # frozenset({s,j}) -> col (PERSISTENT, never reassigned)
        self.spk_partner = {}         # s -> NN(s) for the current epoch
        self.activated_cols = set()   # columns used this epoch (reset each epoch)

    @torch.no_grad()
    def rebuild_pairing(self, W):
        """Recompute global NN pairing from current real prototypes W (D, C).

        Columns are assigned persistently: a pair already in `pair_col` keeps its
        column; only genuinely new pairs consume a fresh column (until max_cols).
        Resets the per-epoch activated set. Identical on all DDP ranks because W
        is synchronised at the epoch boundary.
        """
        Wn = F.normalize(W.detach(), dim=0)          # (D, C)
        cos = Wn.t() @ Wn                            # (C, C)
        cos.fill_diagonal_(-2.0)
        nn_idx = cos.argmax(dim=1).tolist()          # NN per speaker
        self.spk_partner = {s: int(nn_idx[s]) for s in range(self.num_real)}
        for s in range(self.num_real):
            key = self._key(s, self.spk_partner[s])
            if key not in self.pair_col and len(self.pair_col) < self.max_cols:
                self.pair_col[key] = len(self.pair_col)
        self.activated_cols = set()

    @staticmethod
    def _key(s, j):
        return frozenset((int(s), int(j))) if s != j else frozenset((int(s),))

    def col_of(self, s):
        """Return (col, partner) for speaker s under the current epoch pairing.
        Returns (None, None) if the pair has no column (table full / not built)."""
        j = self.spk_partner.get(int(s))
        if j is None:
            return None, None
        col = self.pair_col.get(self._key(s, j))
        return col, j

    @torch.no_grad()
    def update_bank(self, emb_detached, labels):
        """Push current (detached) embeddings into their speakers' queues."""
        for e, l in zip(emb_detached, labels.tolist()):
            self.bank[int(l)].append(e)

    def partner_embedding(self, j, idx_in_batch, x):
        """Embedding for partner j: current batch (with grad) if present, else a
        recent bank embedding (detached). Returns None if unavailable."""
        if j in idx_in_batch:
            return x[idx_in_batch[j][0]]
        if len(self.bank[j]) > 0:
            return self.bank[j][-1].to(x.device)
        return None
