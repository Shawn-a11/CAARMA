import torch


def _slerp(p0, p1, t=0.5, eps=1e-7):
    """Spherical linear interpolation that preserves vector magnitude via
    geometric-mean interpolation of norms.

    Speaker embeddings (and AM-Softmax class prototypes) live on the unit
    hypersphere up to a scalar magnitude — LERP in Euclidean space pulls
    midpoints *into* the sphere and distorts angular relationships, whereas
    SLERP stays on the geodesic between the two unit directions.

    p0, p1: (..., dim) tensors.
    """
    norm0 = p0.norm(dim=-1, keepdim=True).clamp(min=eps)
    norm1 = p1.norm(dim=-1, keepdim=True).clamp(min=eps)
    u0 = p0 / norm0
    u1 = p1 / norm1

    cos_omega = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    # Fall back to LERP when vectors are nearly colinear (sin_omega ~ 0).
    near_linear = sin_omega < eps
    a = torch.where(near_linear, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega.clamp(min=eps))
    b = torch.where(near_linear, torch.full_like(a, t), torch.sin(t * omega) / sin_omega.clamp(min=eps))

    u_mix = a * u0 + b * u1
    # Geometric mean of norms preserves the magnitude scale on the manifold.
    norm_mix = norm0.pow(1.0 - t) * norm1.pow(t)
    return u_mix * norm_mix


def mixup_data_euc_avg(x, W, labels):
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
        # synthetic class (their midpoint is identical -> cos=1 otherwise).
        dictidx = (min(int(l1), int(l2)), max(int(l1), int(l2)))
        if dictidx not in newlabel:
            newlabel[dictidx] = labelid
            pair_l1.append(l1)
            pair_l2.append(l2)
            labelid += 1
        else:
            # Update the existing slot's pair (idempotent: l1, l2 already match).
            slot = newlabel[dictidx]
            pair_l1[slot] = l1
            pair_l2[slot] = l2
        y_mix[i] = newlabel[dictidx]
        index.append(lst_labels.index(l2))

    # SLERP on AM-Softmax prototypes for the unique label pairs.
    if labelid > 0:
        W_l1 = W[:, pair_l1].t()                 # (labelid, W_dim)
        W_l2 = W[:, pair_l2].t()                 # (labelid, W_dim)
        w_pairs = _slerp(W_l1, W_l2, t=0.5)      # (labelid, W_dim)
        w_mix = w_pairs.t().contiguous()         # (W_dim, labelid)
    else:
        w_mix = w_mix[:, :0]

    # SLERP on embeddings — geodesic midpoint between speaker pairs.
    x_mix = _slerp(x, x[index, :], t=0.5)

    x_combined = x_mix
    w_combined = w_mix.to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined, w_combined
