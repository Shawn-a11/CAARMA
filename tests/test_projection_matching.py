import torch
import torch.nn.functional as F

from model.discriminator_mix import ProjectionDiscriminator_spectral


def test_pairwise_compatibility_matches_selected_logits():
    torch.manual_seed(7)
    discriminator = ProjectionDiscriminator_spectral(
        embedding_dim=6, hidden_dim=8
    ).eval()
    embeddings = torch.randn(4, 6)
    conditions = torch.randn(4, 6)
    negative_index = torch.tensor([1, 2, 3, 0])

    pairwise = discriminator.pairwise_compatibility(embeddings, conditions)
    positive, negative = discriminator.matching_logits(
        embeddings, conditions, conditions[negative_index]
    )

    expected_positive = pairwise.diagonal()
    expected_negative = pairwise[
        torch.arange(embeddings.size(0)), negative_index
    ]
    assert torch.allclose(positive, expected_positive, atol=1e-6)
    assert torch.allclose(negative, expected_negative, atol=1e-6)


def test_pairwise_softplus_prefers_larger_match_gap():
    weak_gap = torch.tensor([0.1, -0.2])
    strong_gap = torch.tensor([1.1, 0.8])

    weak_loss = F.softplus(-weak_gap).mean()
    strong_loss = F.softplus(-strong_gap).mean()

    assert strong_loss < weak_loss
