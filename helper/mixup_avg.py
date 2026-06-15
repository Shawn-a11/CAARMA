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
        # Bug-fix (nearest-neighbour index mismatch): the original code argmins
        # over a list that EXCLUDES single_spk, but then indexes set_label (which
        # INCLUDES it) with that filtered position -> off-by-one, so the chosen
        # "nearest" was really (true_NN - 1) and matched the true NN only ~32% of
        # the time (near-random pairing). We build the candidate list explicitly
        # and index INTO IT, so closest_speaker is the genuine nearest neighbour.
        # The self-match fallback is no longer needed (candidates excludes self).
        candidates = [speaker for speaker in set_label if single_spk != speaker]
        distances = [torch.dist(W[:, single_spk], W[:, speaker]) for speaker in candidates]
        closest_neighbor_index = torch.argmin(torch.tensor(distances))
        dic_spk[single_spk] = int(candidates[closest_neighbor_index])
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

