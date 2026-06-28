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

CRP / stick-breaking create-vs-reuse control is the v2 axis: instead of always
using one fixed nearest-neighbour pair per anchor speaker, the state can choose
between reusing an already introduced synthetic pair and creating a new pair
from the anchor's top-k neighbours.
"""

from collections import defaultdict, deque
import math
import random

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

    def __init__(self, num_real, max_cols, bank_size=10, pair_strategy="fixed_nn",
                 crp_alpha=1.0, crp_topk=4, spk_attr=None,
                 attr_constraint="none"):
        self.num_real = int(num_real)
        self.max_cols = int(max_cols)
        self.bank_size = int(bank_size)
        self.pair_strategy = pair_strategy
        self.crp_alpha = float(crp_alpha)
        self.crp_topk = int(crp_topk)
        self.spk_attr = None
        if spk_attr is not None:
            self.spk_attr = torch.as_tensor(spk_attr, dtype=torch.long).cpu()
        self.attr_constraint = attr_constraint or "none"
        self.bank = defaultdict(lambda: deque(maxlen=self.bank_size))  # spk -> recent detached emb (D,)
        self.pair_col = {}            # frozenset({s,j}) -> col (PERSISTENT, never reassigned)
        self.spk_partner = {}         # s -> NN(s) for the current epoch
        self.candidate_pairs = defaultdict(list)  # s -> candidate pair keys
        self.created_pairs = set()     # pairs that have produced at least one synthetic sample
        self.pair_visits = defaultdict(int)
        self.total_pair_visits = 0
        self.activated_cols = set()   # columns used this epoch (reset each epoch)
        self.last_stats = self._empty_stats()

    @torch.no_grad()
    def rebuild_pairing(self, W):
        """Recompute candidate pair sets from current real prototypes W (D, C).

        Columns are assigned persistently: a pair already in `pair_col` keeps its
        column; only genuinely new pairs consume a fresh column (until max_cols).
        Resets the per-epoch activated set. Identical on all DDP ranks because W
        is synchronised at the epoch boundary.
        """
        Wn = F.normalize(W.detach(), dim=0)          # (D, C)
        cos = Wn.t() @ Wn                            # (C, C)
        cos.fill_diagonal_(-2.0)
        k = max(1, min(self.crp_topk, self.num_real - 1))
        topk_idx = []
        for s in range(self.num_real):
            scores = cos[s].clone()
            if self._use_attr_constraint():
                allowed = self._same_attr_mask(s).to(scores.device)
                if allowed.any():
                    scores[~allowed] = -2.0
                # If metadata is missing for this speaker, keep the original
                # unconstrained NN fallback instead of disabling synthesis.
            topk_idx.append(scores.topk(k=k, dim=0).indices.tolist())
        self.spk_partner = {s: int(topk_idx[s][0]) for s in range(self.num_real)}
        self.candidate_pairs = defaultdict(list)

        for s in range(self.num_real):
            partners = [self.spk_partner[s]]
            if self.pair_strategy == "crp":
                partners = [int(j) for j in topk_idx[s]]
            for j in partners:
                key = self._key(s, j)
                if self.pair_strategy != "crp":
                    self._ensure_col(key)
                if key not in self.candidate_pairs[s]:
                    self.candidate_pairs[s].append(key)

        self._rebuild_pairs_by_speaker()
        self.activated_cols = set()
        self.last_stats = self._empty_stats()

    @staticmethod
    def _key(s, j):
        return frozenset((int(s), int(j))) if s != j else frozenset((int(s),))

    @staticmethod
    def _serialise_key(key):
        return tuple(sorted(int(v) for v in key))

    @staticmethod
    def _deserialise_key(values):
        return frozenset(int(v) for v in values)

    @staticmethod
    def _members(key):
        vals = tuple(key)
        if len(vals) == 1:
            return vals[0], vals[0]
        return vals[0], vals[1]

    def _use_attr_constraint(self):
        return self.attr_constraint != "none" and self.spk_attr is not None

    def _same_attr_mask(self, s):
        attr = self.spk_attr
        if attr is None or int(s) >= attr.numel() or int(attr[int(s)]) < 0:
            mask = torch.ones(self.num_real, dtype=torch.bool)
            mask[int(s)] = False
            return mask
        mask = attr[:self.num_real] == attr[int(s)]
        mask[int(s)] = False
        return mask

    def _other(self, key, s):
        a, b = self._members(key)
        return b if int(s) == a else a

    def _ensure_col(self, key):
        if key not in self.pair_col and len(self.pair_col) < self.max_cols:
            self.pair_col[key] = len(self.pair_col)
        return self.pair_col.get(key)

    def _rebuild_pairs_by_speaker(self):
        self.pairs_by_spk = defaultdict(list)
        for key in self.pair_col:
            a, b = self._members(key)
            self.pairs_by_spk[a].append(key)
            if b != a:
                self.pairs_by_spk[b].append(key)

    def col_of(self, s):
        """Return (col, partner) for speaker s under the current epoch pairing.
        Returns (None, None) if the pair has no column (table full / not built)."""
        key, col, j, _ = self.select_pair(s)
        if key is None:
            return None, None
        return col, j

    def select_pair(self, s):
        """Choose a synthetic pair for anchor speaker s.

        fixed_nn: always use the current nearest-neighbour pair.
        crp: choose between reusing an introduced pair containing s and creating
        a new top-k-neighbour pair, with alpha controlling the create mass.
        Returns (key, col, partner, event), where event is "new" or "reuse".
        The caller commits the visit only after a usable partner embedding exists.
        """
        s = int(s)
        if self.pair_strategy != "crp":
            j = self.spk_partner.get(s)
            if j is None:
                return None, None, None, "none"
            key = self._key(s, j)
            col = self._ensure_col(key)
            event = "reuse" if key in self.created_pairs else "new"
            return key, col, j, event

        candidates = list(self.candidate_pairs.get(s, []))
        if not candidates:
            return self._select_assigned_fallback(s)

        assigned = [k for k in candidates if k in self.pair_col]
        existing = [k for k in assigned if k in self.created_pairs]
        novel = [k for k in candidates if k not in self.pair_col]

        if existing and novel:
            local_visits = sum(max(1, self.pair_visits[k]) for k in existing)
            p_new = self.crp_alpha / (local_visits + self.crp_alpha)
            choose_new = random.random() < p_new
        else:
            choose_new = bool(novel)

        if choose_new:
            key = random.choice(novel)
            col = self._ensure_col(key)
            if col is None:
                return self._select_assigned_fallback(s)
            event = "new"
        else:
            if not existing:
                reusable = assigned or self.pairs_by_spk.get(s, [])
                if not reusable:
                    return None, None, None, "none"
                key = random.choice(reusable)
                return key, self.pair_col[key], self._other(key, s), "reuse"
            weights = [max(1, self.pair_visits[k]) for k in existing]
            key = random.choices(existing, weights=weights, k=1)[0]
            event = "reuse"
            col = self.pair_col[key]

        return key, col, self._other(key, s), event

    def _select_assigned_fallback(self, s):
        """Reuse any already assigned pair for s when top-k creation is blocked.

        This keeps training valid after the synthetic table reaches max_cols.
        """
        s = int(s)
        reusable = self.pairs_by_spk.get(s, [])
        if reusable:
            weights = [max(1, self.pair_visits[k]) for k in reusable]
            key = random.choices(reusable, weights=weights, k=1)[0]
            return key, self.pair_col[key], self._other(key, s), "reuse"
        j = self.spk_partner.get(s)
        if j is None:
            return None, None, None, "none"
        key = self._key(s, j)
        col = self._ensure_col(key)
        if col is None:
            return None, None, None, "none"
        event = "reuse" if key in self.created_pairs else "new"
        return key, col, j, event

    def commit_visit(self, key, col, event, source):
        """Record one successful synthetic sample for diagnostics and CRP state."""
        self.created_pairs.add(key)
        self.pair_visits[key] += 1
        self.total_pair_visits += 1
        self.activated_cols.add(int(col))
        self._batch_stats["num_synth"] += 1
        if event == "new":
            self._batch_stats["new"] += 1
        elif event == "reuse":
            self._batch_stats["reuse"] += 1
        if source == "bank":
            self._batch_stats["bank"] += 1
        elif source == "batch":
            self._batch_stats["batch"] += 1

    def begin_batch_stats(self, batch_size):
        self._batch_stats = {
            "batch_size": int(batch_size),
            "num_synth": 0,
            "new": 0,
            "reuse": 0,
            "bank": 0,
            "batch": 0,
        }

    def finalize_batch_stats(self):
        bs = max(1, self._batch_stats["batch_size"])
        ns = max(1, self._batch_stats["num_synth"])
        visits = [self.pair_visits[k] for k in self.created_pairs if self.pair_visits[k] > 0]
        total = float(sum(visits))
        if len(visits) > 1 and total > 0:
            probs = [v / total for v in visits]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(len(probs))
        else:
            entropy = 0.0
        created = max(1, len(self.created_pairs))
        self.last_stats = {
            "synth_table_size": float(len(self.created_pairs)),
            "active_synth_cols": float(len(self.activated_cols)),
            "reuse_rate": float(self._batch_stats["reuse"]) / ns,
            "new_pair_rate": float(self._batch_stats["new"]) / ns,
            "mean_visits_per_pair": float(self.total_pair_visits) / created,
            "pair_visit_entropy": float(entropy),
            "Ns_over_B": float(self._batch_stats["num_synth"]) / bs,
            "bank_hit_rate": float(self._batch_stats["bank"]) / ns,
            "batch_hit_rate": float(self._batch_stats["batch"]) / ns,
        }
        return self.last_stats

    def _empty_stats(self):
        return {
            "synth_table_size": 0.0,
            "active_synth_cols": 0.0,
            "reuse_rate": 0.0,
            "new_pair_rate": 0.0,
            "mean_visits_per_pair": 0.0,
            "pair_visit_entropy": 0.0,
            "Ns_over_B": 0.0,
            "bank_hit_rate": 0.0,
            "batch_hit_rate": 0.0,
        }

    def state_dict(self):
        """Serialise non-module state for Lightning/PyTorch checkpoints.

        The learnable W_syn is saved by the criterion as a normal Parameter.
        This method saves the identity table, visit counts, and memory bank.
        Bank tensors are moved to CPU to keep checkpoints device-agnostic.
        """
        return {
            "num_real": self.num_real,
            "max_cols": self.max_cols,
            "bank_size": self.bank_size,
            "pair_strategy": self.pair_strategy,
            "crp_alpha": self.crp_alpha,
            "crp_topk": self.crp_topk,
            "spk_attr": None if self.spk_attr is None else self.spk_attr.cpu(),
            "attr_constraint": self.attr_constraint,
            "bank": {
                int(spk): [emb.detach().cpu() for emb in queue]
                for spk, queue in self.bank.items()
            },
            "pair_col": [
                (self._serialise_key(key), int(col))
                for key, col in self.pair_col.items()
            ],
            "spk_partner": {int(s): int(j) for s, j in self.spk_partner.items()},
            "candidate_pairs": {
                int(s): [self._serialise_key(key) for key in keys]
                for s, keys in self.candidate_pairs.items()
            },
            "created_pairs": [self._serialise_key(key) for key in self.created_pairs],
            "pair_visits": [
                (self._serialise_key(key), int(count))
                for key, count in self.pair_visits.items()
            ],
            "total_pair_visits": int(self.total_pair_visits),
            "activated_cols": [int(c) for c in self.activated_cols],
            "last_stats": dict(self.last_stats),
        }

    def load_state_dict(self, state):
        """Restore state saved by state_dict().

        Missing fields are tolerated so older checkpoints can still be loaded.
        """
        if not state:
            return

        self.bank_size = int(state.get("bank_size", self.bank_size))
        self.pair_strategy = state.get("pair_strategy", self.pair_strategy)
        self.crp_alpha = float(state.get("crp_alpha", self.crp_alpha))
        self.crp_topk = int(state.get("crp_topk", self.crp_topk))
        self.attr_constraint = state.get("attr_constraint", self.attr_constraint)
        if state.get("spk_attr") is not None:
            self.spk_attr = torch.as_tensor(state["spk_attr"], dtype=torch.long).cpu()

        self.bank = defaultdict(lambda: deque(maxlen=self.bank_size))
        for spk, queue in state.get("bank", {}).items():
            self.bank[int(spk)] = deque(
                [emb.detach().cpu() for emb in queue],
                maxlen=self.bank_size,
            )

        self.pair_col = {}
        for key_values, col in state.get("pair_col", []):
            self.pair_col[self._deserialise_key(key_values)] = int(col)

        self.spk_partner = {
            int(s): int(j) for s, j in state.get("spk_partner", {}).items()
        }

        self.candidate_pairs = defaultdict(list)
        for s, keys in state.get("candidate_pairs", {}).items():
            self.candidate_pairs[int(s)] = [
                self._deserialise_key(key_values) for key_values in keys
            ]

        self.created_pairs = {
            self._deserialise_key(key_values)
            for key_values in state.get("created_pairs", [])
        }

        self.pair_visits = defaultdict(int)
        for key_values, count in state.get("pair_visits", []):
            self.pair_visits[self._deserialise_key(key_values)] = int(count)

        self.total_pair_visits = int(state.get("total_pair_visits", 0))
        self.activated_cols = set(int(c) for c in state.get("activated_cols", []))
        self.last_stats = dict(state.get("last_stats", self._empty_stats()))
        self._rebuild_pairs_by_speaker()

    @torch.no_grad()
    def update_bank(self, emb_detached, labels):
        """Push current (detached) embeddings into their speakers' queues."""
        for e, l in zip(emb_detached, labels.tolist()):
            self.bank[int(l)].append(e)

    def partner_embedding(self, j, idx_in_batch, x):
        """Embedding for partner j: current batch (with grad) if present, else a
        recent bank embedding (detached). Returns (None, None) if unavailable."""
        if j in idx_in_batch:
            return x[idx_in_batch[j][0]], "batch"
        if len(self.bank[j]) > 0:
            return self.bank[j][-1].to(x.device), "bank"
        return None, None
