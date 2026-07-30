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
import torch.distributed as dist
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
                 crp_alpha=1.0, crp_topk=4, reuse_policy="popularity",
                 reuse_power=1.0, candidate_pool="topk", cluster_size=8,
                 cluster_candidate_selection="nearest",
                 cluster_candidate_seed=1729):
        self.num_real = int(num_real)
        self.max_cols = int(max_cols)
        self.bank_size = int(bank_size)
        self.pair_strategy = pair_strategy
        self.crp_alpha = float(crp_alpha)
        self.crp_topk = int(crp_topk)
        self.reuse_policy = str(reuse_policy)
        if self.reuse_policy not in {"popularity", "powered", "fisher_ucb"}:
            raise ValueError(
                "reuse_policy must be 'popularity', 'powered', or 'fisher_ucb'"
            )
        self.reuse_power = float(reuse_power)
        if not 0.0 < self.reuse_power <= 1.0:
            raise ValueError("reuse_power must be in (0, 1]")
        self.candidate_pool = str(candidate_pool)
        self.cluster_size = int(cluster_size)
        self.cluster_candidate_selection = str(cluster_candidate_selection)
        self.cluster_candidate_seed = int(cluster_candidate_seed)
        if self.candidate_pool not in {"topk", "cluster"}:
            raise ValueError("candidate_pool must be 'topk' or 'cluster'")
        if self.cluster_candidate_selection not in {"nearest", "random"}:
            raise ValueError(
                "cluster_candidate_selection must be 'nearest' or 'random'"
            )
        if self.candidate_pool == "cluster" and self.pair_strategy != "crp":
            raise ValueError(
                "candidate_pool='cluster' requires pair_strategy='crp'"
            )
        if self.candidate_pool == "cluster" and self.cluster_size < 2:
            raise ValueError("cluster_size must be >= 2")
        self.cluster_assign = []
        self.bank = defaultdict(lambda: deque(maxlen=self.bank_size))  # spk -> recent detached emb (D,)
        self.pair_col = {}            # frozenset({s,j}) -> col (PERSISTENT, never reassigned)
        self.col_pair = {}            # inverse map used by DDP tensor reductions
        self.spk_partner = {}         # s -> NN(s) for the current epoch
        self.candidate_pairs = defaultdict(list)  # s -> candidate pair keys
        self.created_pairs = set()     # pairs that have produced at least one synthetic sample
        self.pair_visits = defaultdict(int)
        self.total_pair_visits = 0
        self.pair_reward_sum = defaultdict(float)
        self.pair_reward_updates = defaultdict(int)
        self.total_reward_updates = 0
        self.pair_last_p = {}
        self.activated_cols = set()   # columns used this epoch (reset each epoch)
        self._reset_pending()
        self.last_stats = self._empty_stats()

    @staticmethod
    @torch.no_grad()
    def _spherical_kmeans(points, num_clusters, iters=20, seed=0):
        """Deterministic cosine k-means over unit prototype rows."""
        points = F.normalize(
            points.detach().to(dtype=torch.float32, device="cpu"), dim=1
        )
        num_points = points.size(0)
        num_clusters = max(1, min(int(num_clusters), num_points))
        generator = torch.Generator().manual_seed(int(seed))

        first = int(torch.randint(num_points, (1,), generator=generator))
        centers = [points[first]]
        for _ in range(1, num_clusters):
            similarity = torch.stack(
                [points @ center for center in centers], dim=1
            ).max(dim=1).values
            distance = (1.0 - similarity).clamp(min=0.0)
            total = float(distance.sum())
            probabilities = (
                distance / total
                if total > 0
                else torch.full((num_points,), 1.0 / num_points)
            )
            index = int(torch.multinomial(
                probabilities, 1, generator=generator
            ))
            centers.append(points[index])
        centers = torch.stack(centers, dim=0)

        assignment = None
        for _ in range(max(1, int(iters))):
            similarity = points @ F.normalize(centers, dim=1).t()
            new_assignment = similarity.argmax(dim=1)
            if assignment is not None and torch.equal(
                    new_assignment, assignment):
                break
            assignment = new_assignment
            for cluster in range(num_clusters):
                members = points[assignment == cluster]
                if members.size(0) == 0:
                    served = similarity.max(dim=1).values
                    centers[cluster] = points[int(served.argmin())]
                else:
                    centers[cluster] = members.mean(dim=0)
        return [int(cluster) for cluster in assignment.tolist()]

    def _cluster_candidates(self, prototype_rows, cosine):
        """Return k candidates from each anchor's natural cluster.

        ``nearest`` is the E3 method: rank same-cluster speakers by cosine.
        ``random`` is the matched ablation: sample the same number of candidates
        uniformly without replacement, removing only the within-cluster
        nearest-neighbour ranking. Per-anchor seeds keep all DDP ranks aligned.
        """
        num_clusters = max(1, round(self.num_real / self.cluster_size))
        self.cluster_assign = self._spherical_kmeans(
            prototype_rows, num_clusters
        )
        members = defaultdict(list)
        for speaker, cluster in enumerate(self.cluster_assign):
            members[cluster].append(speaker)

        k = max(1, min(self.crp_topk, self.num_real - 1))
        candidates = {}
        for speaker in range(self.num_real):
            neighbours = [
                other for other in members[self.cluster_assign[speaker]]
                if other != speaker
            ]
            if neighbours:
                if self.cluster_candidate_selection == "nearest":
                    neighbours.sort(
                        key=lambda other: float(cosine[speaker][other]),
                        reverse=True,
                    )
                else:
                    generator = torch.Generator().manual_seed(
                        self.cluster_candidate_seed + speaker
                    )
                    order = torch.randperm(
                        len(neighbours), generator=generator
                    ).tolist()
                    neighbours = [neighbours[index] for index in order]
                candidates[speaker] = neighbours[:k]
            else:
                candidates[speaker] = [self.spk_partner[speaker]]
        return candidates

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
        topk_idx = cos.topk(k=k, dim=1).indices.tolist()
        self.spk_partner = {s: int(topk_idx[s][0]) for s in range(self.num_real)}
        self.candidate_pairs = defaultdict(list)

        # Columns reserved for candidates that were never visited carry no
        # learned pseudo-class identity and can be safely recycled. Created
        # classes keep their columns permanently across epochs.
        self.pair_col = {
            key: col for key, col in self.pair_col.items()
            if key in self.created_pairs
        }
        self.col_pair = {col: key for key, col in self.pair_col.items()}

        if self.pair_strategy == "crp" and self.candidate_pool == "cluster":
            per_anchor = self._cluster_candidates(Wn.t(), cos)
        else:
            per_anchor = None
            self.cluster_assign = []

        all_candidate_keys = set()
        for s in range(self.num_real):
            partners = [self.spk_partner[s]]
            if self.pair_strategy == "crp":
                partners = (
                    [int(j) for j in per_anchor[s]]
                    if per_anchor is not None
                    else [int(j) for j in topk_idx[s]]
                )
            for j in partners:
                key = self._key(s, j)
                all_candidate_keys.add(key)
                if key not in self.candidate_pairs[s]:
                    self.candidate_pairs[s].append(key)

        # Reserve pair columns in a deterministic global order. This does not
        # create a CRP class; creation still happens on its first committed
        # visit. It only guarantees that every DDP rank gives a pair the same
        # W_syn column even when the ranks observe different local speakers.
        free_cols = iter(sorted(set(range(self.max_cols)) - set(self.col_pair)))
        newly_reserved = []
        for key in sorted(all_candidate_keys, key=self._serialise_key):
            if key in self.pair_col:
                continue
            try:
                col = next(free_cols)
            except StopIteration:
                break
            self.pair_col[key] = col
            self.col_pair[col] = key
            newly_reserved.append((key, col))

        self._rebuild_pairs_by_speaker()
        self.activated_cols = set()
        self.last_stats = self._empty_stats()
        return newly_reserved

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

    def _other(self, key, s):
        a, b = self._members(key)
        return b if int(s) == a else a

    def _ensure_col(self, key):
        if key not in self.pair_col and len(self.pair_col) < self.max_cols:
            used = set(self.pair_col.values())
            col = next(c for c in range(self.max_cols) if c not in used)
            self.pair_col[key] = col
            self.col_pair[col] = key
        return self.pair_col.get(key)

    def _rebuild_pairs_by_speaker(self):
        self.pairs_by_spk = defaultdict(list)
        self.col_pair = {}
        for key in self.pair_col:
            self.col_pair[self.pair_col[key]] = key
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

    def _ucb_score(self, key):
        """Standard UCB1 score for one persistent synthetic class."""
        n = self.pair_reward_updates[key]
        if n <= 0:
            return float("inf")
        total_updates = max(2, self.total_reward_updates)
        mean_reward = self.pair_reward_sum[key] / n
        return mean_reward + math.sqrt(2.0 * math.log(total_updates) / n)

    def _select_reusable(self, keys):
        if not keys:
            return None
        if self.reuse_policy == "fisher_ucb":
            # Stable tie breaks make the policy reproducible and avoid adding a
            # second exploration hyperparameter on top of standard UCB1.
            return min(
                keys,
                key=lambda key: (
                    -self._ucb_score(key),
                    self.pair_visits[key],
                    self._serialise_key(key),
                ),
            )
        weights = [self._reuse_mass(key) for key in keys]
        return random.choices(keys, weights=weights, k=1)[0]

    def _reuse_mass(self, key):
        """Occupancy mass used by create-vs-reuse and reusable-class sampling."""
        visits = float(max(1, self.pair_visits[key]))
        if self.reuse_policy == "powered":
            return visits ** self.reuse_power
        return visits

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
        novel = [k for k in assigned if k not in self.created_pairs]

        if existing and novel:
            existing_mass = sum(self._reuse_mass(key) for key in existing)
            p_new = self.crp_alpha / (existing_mass + self.crp_alpha)
            choose_new = random.random() < p_new
        else:
            choose_new = bool(novel)

        if choose_new:
            key = random.choice(novel)
            col = self.pair_col[key]
            event = "new"
        else:
            if not existing:
                reusable = assigned or self.pairs_by_spk.get(s, [])
                if not reusable:
                    return None, None, None, "none"
                key = random.choice(reusable)
                return key, self.pair_col[key], self._other(key, s), "reuse"
            key = self._select_reusable(existing)
            event = "reuse"
            col = self.pair_col[key]

        return key, col, self._other(key, s), event

    def _select_assigned_fallback(self, s):
        """Reuse any already assigned pair for s when top-k creation is blocked.

        This keeps training valid after the synthetic table reaches max_cols.
        """
        s = int(s)
        reusable = [
            key for key in self.pairs_by_spk.get(s, [])
            if key in self.created_pairs
        ]
        if reusable:
            key = self._select_reusable(reusable)
            return key, self.pair_col[key], self._other(key, s), "reuse"
        j = self.spk_partner.get(s)
        if j is None:
            return None, None, None, "none"
        key = self._key(s, j)
        col = self.pair_col.get(key)
        if col is None:
            return None, None, None, "none"
        event = "reuse" if key in self.created_pairs else "new"
        return key, col, j, event

    def commit_visit(self, key, col, event, source):
        """Stage one successful visit; global state is committed after DDP sync."""
        self._pending_visits[int(col)] += 1
        self._pending_activated.add(int(col))
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
        self._reset_pending()
        self._batch_stats = {
            "batch_size": int(batch_size),
            "num_synth": 0,
            "new": 0,
            "reuse": 0,
            "bank": 0,
            "batch": 0,
            "utility_sum": 0.0,
            "utility_count": 0,
        }

    def _reset_pending(self):
        self._pending_visits = defaultdict(int)
        self._pending_reward_sum = defaultdict(float)
        self._pending_reward_count = defaultdict(int)
        self._pending_p_sum = defaultdict(float)
        self._pending_p_count = defaultdict(int)
        self._pending_activated = set()

    @torch.no_grad()
    def record_utilities(self, cols, probabilities, fisher_utilities):
        """Stage bounded class-level rewards from the current joint-L_syn logits.

        One arm pull is one class exposure on one rank. Repeated samples of the
        same class in a batch are averaged before the Fisher utility and positive
        learning-progress signal are combined.
        """
        grouped_p = defaultdict(list)
        grouped_u = defaultdict(list)
        for col, probability, utility in zip(
                cols, probabilities.detach().cpu().tolist(),
                fisher_utilities.detach().cpu().tolist()):
            grouped_p[int(col)].append(float(probability))
            grouped_u[int(col)].append(float(utility))

        for col, values in grouped_p.items():
            key = self.col_pair.get(col)
            if key is None:
                continue
            mean_p = sum(values) / len(values)
            mean_fisher = sum(grouped_u[col]) / len(grouped_u[col])
            previous_p = self.pair_last_p.get(key)
            progress = 0.0 if previous_p is None else max(0.0, mean_p - previous_p)
            reward = max(mean_fisher, progress)
            reward = min(1.0, max(0.0, reward))

            self._pending_reward_sum[col] += reward
            self._pending_reward_count[col] += 1
            self._pending_p_sum[col] += mean_p
            self._pending_p_count[col] += 1
            self._batch_stats["utility_sum"] += reward
            self._batch_stats["utility_count"] += 1

    @torch.no_grad()
    def synchronize_pending(self, device):
        """All-reduce visit/reward deltas so every DDP rank has one CRP state."""
        pending = torch.zeros((6, self.max_cols), dtype=torch.float64, device=device)
        for col, count in self._pending_visits.items():
            pending[0, col] = count
        for col, value in self._pending_reward_sum.items():
            pending[1, col] = value
        for col, count in self._pending_reward_count.items():
            pending[2, col] = count
        for col, value in self._pending_p_sum.items():
            pending[3, col] = value
        for col, count in self._pending_p_count.items():
            pending[4, col] = count
        for col in self._pending_activated:
            pending[5, col] = 1.0

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pending, op=dist.ReduceOp.SUM)

        changed = torch.nonzero(pending.abs().sum(dim=0) > 0, as_tuple=False)
        for col in changed.flatten().cpu().tolist():
            key = self.col_pair.get(int(col))
            if key is None:
                continue
            visits = int(round(pending[0, col].item()))
            if visits > 0:
                self.created_pairs.add(key)
                self.pair_visits[key] += visits
                self.total_pair_visits += visits
            reward_count = int(round(pending[2, col].item()))
            if reward_count > 0:
                self.pair_reward_sum[key] += pending[1, col].item()
                self.pair_reward_updates[key] += reward_count
                self.total_reward_updates += reward_count
            p_count = int(round(pending[4, col].item()))
            if p_count > 0:
                self.pair_last_p[key] = pending[3, col].item() / p_count
            if pending[5, col].item() > 0:
                self.activated_cols.add(int(col))

        self._reset_pending()
        return self.finalize_batch_stats()

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
        utility_count = max(1, self._batch_stats["utility_count"])
        scored = [
            self._ucb_score(key) for key in self.created_pairs
            if self.pair_reward_updates[key] > 0
        ]
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
            "mean_fisher_ucb_reward": self._batch_stats["utility_sum"] / utility_count,
            "mean_ucb_score": sum(scored) / max(1, len(scored)),
            "utility_coverage": float(len(scored)) / created,
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
            "mean_fisher_ucb_reward": 0.0,
            "mean_ucb_score": 0.0,
            "utility_coverage": 0.0,
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
            "reuse_policy": self.reuse_policy,
            "reuse_power": self.reuse_power,
            "candidate_pool": self.candidate_pool,
            "cluster_size": self.cluster_size,
            "cluster_candidate_selection": self.cluster_candidate_selection,
            "cluster_candidate_seed": self.cluster_candidate_seed,
            "cluster_assign": [int(cluster) for cluster in self.cluster_assign],
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
            "pair_reward_sum": [
                (self._serialise_key(key), float(value))
                for key, value in self.pair_reward_sum.items()
            ],
            "pair_reward_updates": [
                (self._serialise_key(key), int(value))
                for key, value in self.pair_reward_updates.items()
            ],
            "total_reward_updates": int(self.total_reward_updates),
            "pair_last_p": [
                (self._serialise_key(key), float(value))
                for key, value in self.pair_last_p.items()
            ],
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
        self.reuse_policy = state.get("reuse_policy", self.reuse_policy)
        self.reuse_power = float(state.get("reuse_power", self.reuse_power))
        self.candidate_pool = state.get(
            "candidate_pool", self.candidate_pool
        )
        self.cluster_size = int(state.get("cluster_size", self.cluster_size))
        self.cluster_candidate_selection = state.get(
            "cluster_candidate_selection", self.cluster_candidate_selection
        )
        self.cluster_candidate_seed = int(state.get(
            "cluster_candidate_seed", self.cluster_candidate_seed
        ))
        self.cluster_assign = [
            int(cluster) for cluster in state.get("cluster_assign", [])
        ]

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
        self.pair_reward_sum = defaultdict(float)
        for key_values, value in state.get("pair_reward_sum", []):
            self.pair_reward_sum[self._deserialise_key(key_values)] = float(value)
        self.pair_reward_updates = defaultdict(int)
        for key_values, value in state.get("pair_reward_updates", []):
            self.pair_reward_updates[self._deserialise_key(key_values)] = int(value)
        self.total_reward_updates = int(
            state.get("total_reward_updates", sum(self.pair_reward_updates.values()))
        )
        self.pair_last_p = {}
        for key_values, value in state.get("pair_last_p", []):
            self.pair_last_p[self._deserialise_key(key_values)] = float(value)
        self.activated_cols = set(int(c) for c in state.get("activated_cols", []))
        self.last_stats = dict(state.get("last_stats", self._empty_stats()))
        self._reset_pending()
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
