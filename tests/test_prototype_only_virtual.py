import torch

from criterion.amsoftmax_mix_gan import amsoftmax_gan


def test_virtual_prototypes_are_denominator_only():
    torch.manual_seed(7)
    loss_fn = amsoftmax_gan(
        embedding_dim=4,
        num_classes=6,
        prototype_only_virtual=True,
        virtual_negative_topk=2,
        virtual_negative_t=0.5,
        virtual_negatives_per_batch=2,
        scale=1,
    )
    x = torch.randn(4, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    virtual = loss_fn._build_virtual_negative_weights(labels)
    assert virtual.shape == (4, 2)
    assert not virtual.requires_grad

    base_loss, _ = loss_fn._real_amsoftmax_loss(x, labels)
    virtual_loss, _ = loss_fn._real_amsoftmax_loss(x, labels, virtual)
    assert virtual_loss > base_loss

    loss, _, synthetic = loss_fn(x, labels)
    assert synthetic.shape == (0, 4)
    assert loss_fn.last_virtual_count == 2
    assert torch.isfinite(loss)

    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert loss_fn.W.grad is not None and torch.isfinite(loss_fn.W.grad).all()


def test_synthetic_positive_path_is_rejected():
    loss_fn = amsoftmax_gan(
        embedding_dim=4,
        num_classes=6,
        prototype_only_virtual=True,
    )
    x = torch.randn(2, 4)
    labels = torch.tensor([0, 1])

    try:
        loss_fn(x, labels, flagSyn=True)
    except RuntimeError:
        return
    raise AssertionError('flagSyn=True must fail in prototype-only mode')


def test_zero_budget_is_a_matched_real_only_control():
    loss_fn = amsoftmax_gan(
        embedding_dim=4,
        num_classes=6,
        prototype_only_virtual=True,
        virtual_negatives_per_batch=0,
    )
    x = torch.randn(2, 4)
    labels = torch.tensor([0, 1])

    loss, _, synthetic = loss_fn(x, labels)
    assert torch.isfinite(loss)
    assert loss_fn.last_virtual_count == 0
    assert synthetic.shape == (0, 4)
