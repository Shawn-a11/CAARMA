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

# vMF concentration: higher = tighter around the SLERP midpoint, lower = more
# spread. 50 keeps samples within ~10° of the midpoint for typical embedding
# dimensions, giving meaningful diversity without leaving the class-pair
# manifold neighborhood.
VMF_KAPPA = 50.0


def mixup_data_euc_avg(x, W, labels):
    """Innovation: vMF-sampled spherical mixup.

    Pipeline: nearest-neighbor pair → SLERP midpoint on the unit hypersphere
    → sample from vMF(midpoint, kappa) for the synthetic *embedding*. Class
    prototypes (W) stay at the deterministic SLERP midpoint so AM-Softmax
    anchors remain stable while per-sample synthetic embeddings carry
    controlled stochastic diversity for L_syn and the adversarial loss.
    """
    batch_size = x.size()[0]
    index = []
    w_mix = torch.zeros(W.size(0), batch_size)
    y_mix = torch.zeros(batch_size, dtype=torch.int64)
    set_label = list(set(labels.cpu().detach().numpy()))
    dic_spk = {}
    for single_spk in set_label:
        distances = [torch.dist(W[:, single_spk], W[:, speaker]) for speaker in set_label if single_spk != speaker]
        closest_neighbor_index = torch.argmin(torch.tensor(distances))
        closest_speaker = set_label[closest_neighbor_index]
        if single_spk == closest_speaker.item():
            sorted_distances, sorted_indices = torch.sort(torch.tensor(distances))
            second_closest_neighbor_index = sorted_indices[1]
            second_closest_speaker = set_label[second_closest_neighbor_index]
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
        dictidx = int(str(int(l1)) + str(int(l2)))
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
    x_mix = _sample_vmf(midpoint_x, kappa=VMF_KAPPA)

    x_combined = x_mix
    w_combined = w_mix.to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined, w_combined
