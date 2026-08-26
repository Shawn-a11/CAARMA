import torch


def mixup_data_euc_avg(x, W, labels):
    """Build one-shot synthetic classes from true nearest speaker pairs.

    Pair selection is discrete and therefore uses detached classifier weights.
    The mixed prototypes themselves retain gradients through ``W``.
    """

    if x.ndim != 2 or W.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected x/W to be matrices and labels to be a vector")
    if x.size(0) != labels.numel():
        raise ValueError("Embedding and label batch sizes must match")
    if x.device != W.device:
        raise ValueError("Embeddings and classifier weights must share a device")

    labels = labels.to(device=x.device, dtype=torch.long)
    unique_labels, label_to_unique = torch.unique(
        labels, sorted=True, return_inverse=True
    )
    if unique_labels.numel() < 2:
        raise ValueError("Synthetic mixup requires at least two speakers per batch")

    # Algorithm 1 selects the nearest speaker from the self-excluded label set.
    # Keeping that candidate set in matrix form avoids the original off-by-one
    # error, where an index into the filtered list was applied to the full list.
    with torch.no_grad():
        prototypes = W.detach().index_select(1, unique_labels).transpose(0, 1)
        distances = torch.cdist(prototypes.float(), prototypes.float(), p=2)
        distances.fill_diagonal_(float("inf"))
        nearest_unique = distances.argmin(dim=1)
        nearest_labels = unique_labels.index_select(0, nearest_unique)

    partner_labels = nearest_labels.index_select(0, label_to_unique)
    partner_matches = partner_labels.unsqueeze(1).eq(labels.unsqueeze(0))
    if not bool(partner_matches.any(dim=1).all()):
        raise RuntimeError("A selected synthetic partner is absent from the batch")
    partner_indices = partner_matches.to(torch.int64).argmax(dim=1)
    synthetic_embeddings = 0.5 * (
        x + x.index_select(0, partner_indices)
    )

    # Interpolation is symmetric: (i, j) and (j, i) are one synthetic class.
    # Tensor pair keys also avoid decimal-concatenation collisions such as
    # (1, 23) versus (12, 3).
    pair_labels = torch.stack(
        (
            torch.minimum(labels, partner_labels),
            torch.maximum(labels, partner_labels),
        ),
        dim=1,
    )
    unique_pairs, synthetic_labels = torch.unique(
        pair_labels, dim=0, sorted=True, return_inverse=True
    )
    synthetic_weights = 0.5 * (
        W.index_select(1, unique_pairs[:, 0])
        + W.index_select(1, unique_pairs[:, 1])
    )

    return synthetic_embeddings, synthetic_labels, synthetic_weights
