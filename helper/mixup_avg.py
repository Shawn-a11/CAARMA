import torch


def mixup_data_euc_avg(x, W, labels, spk_attr=None):
    """Nearest-neighbour speaker mixup.

    For each speaker in the batch, pair it with its nearest neighbour (by W
    prototype distance) and blend their embeddings / prototypes to form a
    synthetic speaker.

    spk_attr: optional LongTensor (num_spk,) mapping speaker label -> attribute
    code (gender / nationality). When given, the nearest-neighbour search is
    restricted to speakers sharing the same attribute, so a synthetic speaker
    is only ever a blend of (e.g.) two males or two same-nationality speakers.
    Labels with code -1, or speakers with no same-attribute neighbour in the
    current batch, fall back to the unconstrained nearest neighbour.
    """
    batch_size = x.size()[0]
    index = []
    w_mix = torch.zeros(W.size(0), batch_size)
    y_mix = torch.zeros(batch_size, dtype=torch.int64)
    set_label = list(set(labels.cpu().detach().numpy()))

    dic_spk = {}
    for single_spk in set_label:
        others = [s for s in set_label if s != single_spk]
        if not others:
            dic_spk[single_spk] = single_spk  # alone in batch (degenerate)
            continue

        cand = others
        if spk_attr is not None:
            a = int(spk_attr[int(single_spk)].item())
            if a >= 0:
                same = [s for s in others if int(spk_attr[int(s)].item()) == a]
                if same:                       # keep constraint only if satisfiable
                    cand = same
        # nearest neighbour within the candidate set
        dists = torch.tensor([torch.dist(W[:, single_spk], W[:, s]) for s in cand])
        dic_spk[single_spk] = int(cand[int(torch.argmin(dists))])

    lst_labels = labels.tolist()
    newlabel = {}
    labelid = 0
    for i in range(batch_size):
        l1 = labels[i].item()
        l2 = dic_spk[l1]
        dictidx = int(str(int(l1)) + str(int(l2)))
        if dictidx not in newlabel:
            newlabel[dictidx] = labelid
            w_mix[:, labelid] = (W[:, l1] + W[:, l2]) / 2
            labelid = labelid + 1
        else:
            w_mix[:, newlabel[dictidx]] = (W[:, l1] + W[:, l2]) / 2
        y_mix[i] = newlabel[dictidx]
        index.append(lst_labels.index(l2))
    x_mix = 0.5 * (x + x[index, :])

    x_combined = x_mix
    w_combined = w_mix[:, 0:labelid].to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined, w_combined
