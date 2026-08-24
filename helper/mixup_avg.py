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
    if near.any():
        lerp = F.normalize((1.0 - t) * p0n + t * p1n, dim=-1, eps=eps)
        out = torch.where(near, lerp, out)
    return F.normalize(out, dim=-1, eps=eps)


def mixup_data_euc_avg(x, W, labels):
    batch_size = x.size()[0]
    labelmix = torch.zeros(batch_size, dtype=torch.int64)
    index = []
    w_mix = torch.zeros(W.size(0), batch_size)
    y_mix = torch.zeros(batch_size, dtype=torch.int64)
    set_label = list(set(labels.cpu().detach().numpy()))
    weight_vectors = [W[:, speaker] for speaker in set_label]
    dic_spk = {}
    distances = {}
    for single_spk in set_label:
        distances = [torch.dist(W[:, single_spk], W[:,speaker]) for speaker in set_label if single_spk != speaker]
        closest_neighbor_index = torch.argmin(torch.tensor(distances))
        closest_speaker = set_label[closest_neighbor_index]
        if single_spk == closest_speaker.item():
            sorted_distances, sorted_indices = torch.sort(torch.tensor(distances))
            # Get the second minimum distance and its corresponding speaker
            second_min_distance = sorted_distances[1]
            second_closest_neighbor_index = sorted_indices[1]
            second_closest_speaker = set_label[second_closest_neighbor_index]
            dic_spk[single_spk] = second_closest_speaker.item()
        else:
            dic_spk[single_spk] = closest_speaker.item()
    lst_labels = labels.tolist()
    newlabel = {}
    labelid = 0
    for i in range(batch_size):
        l1 = labels[i].item()
        l2 = dic_spk[l1]
        dictidx = int(str(int(l1)) + str(int(l2)))
        if dictidx not in newlabel:
            newlabel[dictidx] = labelid
            w_mix[:,labelid] = slerp(W[:, l1], W[:, l2], t=0.5)
            labelid = labelid + 1
        else:
            w_mix[:,newlabel[dictidx]] = slerp(W[:, l1], W[:, l2], t=0.5)
        y_mix[i] = newlabel[dictidx]
        index.append(lst_labels.index(l2))
    x_mix = slerp(x, x[index,:], t=0.5)

    x_combined = x_mix
    w_combined = w_mix[:, 0:labelid].to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined , w_combined
