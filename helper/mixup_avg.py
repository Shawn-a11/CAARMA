import torch 

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
    for i in range(batch_size):
        l1 = labels[i].item()
        l2 = dic_spk[l1]
        # De-duplicate synthetic classes: use an UNORDERED pair key so that
        # mutual nearest neighbours (i<->j) map (i,j) and (j,i) to the SAME
        # synthetic class. The old ordered key int(str(l1)+str(l2)) created two
        # classes with identical prototypes (cos=1), which L_syn (a syn-vs-syn
        # softmax) cannot separate -> persistent loss floor + gradient noise.
        # (Also fixes the old key's accidental collisions, e.g. (1,25) vs (12,5).)
        dictidx = (min(int(l1), int(l2)), max(int(l1), int(l2)))
        if dictidx not in newlabel:
            newlabel[dictidx] = labelid
            w_mix[:,labelid] = (W[:, l1] + W[:, l2])/2
            labelid = labelid + 1
        else:
            w_mix[:,newlabel[dictidx]] = (W[:, l1] + W[:, l2])/2
        y_mix[i] = newlabel[dictidx]
        index.append(lst_labels.index(l2))
    x_mix = 0.5*(x + x[index,:])
    
    x_combined = x_mix
    w_combined = w_mix[:, 0:labelid].to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined , w_combined

