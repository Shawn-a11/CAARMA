import math
import torch


# ---------- spherical interpolation primitive ---------- #

def _slerp(p0, p1, t=0.5, eps=1e-7):
    """Spherical linear interpolation that preserves vector magnitude via
    geometric-mean interpolation of norms. p0, p1: (..., dim)."""
    norm0 = p0.norm(dim=-1, keepdim=True).clamp(min=eps)
    norm1 = p1.norm(dim=-1, keepdim=True).clamp(min=eps)
    u0 = p0 / norm0
    u1 = p1 / norm1

    cos_omega = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    near_linear = sin_omega < eps
    a = torch.where(near_linear, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega.clamp(min=eps))
    b = torch.where(near_linear, torch.full_like(a, t), torch.sin(t * omega) / sin_omega.clamp(min=eps))

    u_mix = a * u0 + b * u1
    norm_mix = norm0.pow(1.0 - t) * norm1.pow(t)
    return u_mix * norm_mix


# ---------- vMF sampling (Wood's 1994 rejection algorithm, batched) ---------- #

def _sample_w_vmf(kappa, dim, shape, device, dtype, eps=1e-7, max_iter=20):
    """Sample w = mu^T x where x ~ vMF(mu, kappa). Returns shape `shape`."""
    if dim < 2:
        return torch.ones(shape, device=device, dtype=dtype)

    b = (-2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (dim - 1) ** 2)) / (dim - 1)
    a = ((dim - 1) + 2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (dim - 1) ** 2)) / 4.0
    d = 4.0 * a * b / (1.0 + b) - (dim - 1) * math.log(dim - 1)

    half = (dim - 1) / 2.0
    beta_dist = torch.distributions.Beta(half, half)

    w = torch.zeros(shape, device=device, dtype=dtype)
    pending = torch.ones(shape, device=device, dtype=torch.bool)
    for _ in range(max_iter):
        if not pending.any():
            break
        z = beta_dist.sample(shape).to(device=device, dtype=dtype)
        w_cand = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = torch.rand(shape, device=device, dtype=dtype)
        log_term = (dim - 1) * torch.log((1.0 - z * w_cand).clamp(min=eps)) + kappa * w_cand - d
        accept = log_term >= torch.log(u.clamp(min=eps))
        take = pending & accept
        w = torch.where(take, w_cand, w)
        pending = pending & ~accept
    return w


def _sample_vmf(mu, kappa, eps=1e-7):
    """Sample one point from vMF(mu, kappa). mu: (..., dim); kappa: float.
    Magnitude of mu is preserved (sampled direction × |mu|).
    """
    dim = mu.shape[-1]
    if dim < 2 or kappa <= 0:
        return mu

    norm_mu = mu.norm(dim=-1, keepdim=True).clamp(min=eps)
    direction = mu / norm_mu

    w = _sample_w_vmf(
        kappa, dim,
        shape=mu.shape[:-1],
        device=mu.device, dtype=mu.dtype, eps=eps,
    ).unsqueeze(-1)

    v = torch.randn_like(mu)
    v = v - (v * direction).sum(-1, keepdim=True) * direction
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=eps)

    sampled_dir = w * direction + torch.sqrt((1.0 - w * w).clamp(min=0.0)) * v
    return sampled_dir * norm_mu


# ---------- public mixup interface ---------- #

# vMF concentration parameter.
#
# The "tightness" of vMF(μ, κ) at dimension d is governed by the mean
# resultant length A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ), NOT by the absolute
# value of κ. For large κ (κ ≫ d), A_d(κ) ≈ 1 - (d-1)/(2κ), so the effective
# angular radius is θ_eff ≈ arccos(A_d(κ)).
#
# With our embedding dim d=192:
#   κ=50   → A ≈ 0.18  → ~80°   (≈ uniform on sphere, "vMF" is meaningless)
#   κ=200  → A ≈ 0.55  → ~57°   (loose neighborhood)
#   κ=500  → A ≈ 0.81  → ~36°   (sensible "around midpoint" diversity)
#   κ=1000 → A ≈ 0.91  → ~25°   (≈ real VoxCeleb intra-speaker spread; PSDA-fit)
#   κ=2000 → A ≈ 0.95  → ~18°   (tight)
#
# Default 500 is the recommended starting point; override via config.yaml
# `vmf_kappa` for ablation over {200, 500, 1000, 2000}.
VMF_KAPPA = 500.0


def vmf_effective_angle_deg(kappa: float, dim: int) -> float:
    """Returns the effective angular radius θ_eff (degrees) under the
    large-κ approximation A_d(κ) ≈ 1 - (d-1)/(2κ). For κ < (d-1)/2 the
    distribution is effectively uniform; we return 90° in that case.
    """
    import math
    if kappa <= (dim - 1) / 2.0:
        return 90.0
    a = 1.0 - (dim - 1) / (2.0 * kappa)
    return math.degrees(math.acos(max(-1.0, min(1.0, a))))


def mixup_data_euc_avg(x, W, labels, kappa=None):
    """Innovation: vMF-sampled spherical mixup.

    Pipeline: nearest-neighbor pair → SLERP midpoint on the unit hypersphere
    → sample from vMF(midpoint, kappa) for the synthetic *embedding*. Class
    prototypes (W) stay at the deterministic SLERP midpoint so AM-Softmax
    anchors remain stable while per-sample synthetic embeddings carry
    controlled stochastic diversity for L_syn and the adversarial loss.

    `kappa` defaults to module-level VMF_KAPPA (500.0) but may be overridden
    by the caller (e.g. from a YAML config) for ablation studies.
    """
    if kappa is None:
        kappa = VMF_KAPPA
    batch_size = x.size()[0]
    index = []
    w_mix = torch.zeros(W.size(0), batch_size)
    y_mix = torch.zeros(batch_size, dtype=torch.int64)
    set_label = list(set(labels.cpu().detach().numpy()))
    dic_spk = {}
    for single_spk in set_label:
        # Bug A fix: distances is built over a self-EXCLUDED list, so argmin's
        # index must index that SAME list (`candidates`), not set_label (which
        # includes self) -> the old set_label[...] was off-by-one (matched the
        # true NN only ~half the time). Only the indexed list changes; the
        # fallback is kept as a harmless guard (never fires now: candidates
        # excludes self).
        candidates = [speaker for speaker in set_label if single_spk != speaker]
        distances = [torch.dist(W[:, single_spk], W[:, speaker]) for speaker in candidates]
        closest_neighbor_index = torch.argmin(torch.tensor(distances))
        closest_speaker = candidates[closest_neighbor_index]
        if single_spk == closest_speaker.item():
            sorted_distances, sorted_indices = torch.sort(torch.tensor(distances))
            second_closest_neighbor_index = sorted_indices[1]
            second_closest_speaker = candidates[second_closest_neighbor_index]
            dic_spk[single_spk] = second_closest_speaker.item()
        else:
            dic_spk[single_spk] = closest_speaker.item()

    lst_labels = labels.tolist()
    newlabel = {}
    labelid = 0
    pair_l1, pair_l2 = [], []
    for i in range(batch_size):
        l1 = labels[i].item()
        l2 = dic_spk[l1]
        # Bug B fix: unordered pair key merges mutual NN (i,j)&(j,i) into ONE
        # synthetic class (their SLERP midpoint is identical -> cos=1 otherwise).
        dictidx = (min(int(l1), int(l2)), max(int(l1), int(l2)))
        if dictidx not in newlabel:
            newlabel[dictidx] = labelid
            pair_l1.append(l1)
            pair_l2.append(l2)
            labelid += 1
        else:
            slot = newlabel[dictidx]
            pair_l1[slot] = l1
            pair_l2[slot] = l2
        y_mix[i] = newlabel[dictidx]
        index.append(lst_labels.index(l2))

    # AM-Softmax prototypes: deterministic SLERP midpoint (stable class anchor).
    if labelid > 0:
        W_l1 = W[:, pair_l1].t()
        W_l2 = W[:, pair_l2].t()
        w_pairs = _slerp(W_l1, W_l2, t=0.5)
        w_mix = w_pairs.t().contiguous()
    else:
        w_mix = w_mix[:, :0]

    # Embeddings: SLERP midpoint, then stochastic vMF sample around it.
    midpoint_x = _slerp(x, x[index, :], t=0.5)
    x_mix = _sample_vmf(midpoint_x, kappa=kappa)

    x_combined = x_mix
    w_combined = w_mix.to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined, w_combined
