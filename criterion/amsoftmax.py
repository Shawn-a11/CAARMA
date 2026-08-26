import torch
import torch.nn as nn

from .utils import accuracy


class amsoftmax(nn.Module):
    """AM-Softmax classifier used by the paper's MFA-Conformer baseline."""

    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30, **kwargs):
        super().__init__()
        self.m = float(margin)
        self.s = float(scale)
        self.in_feats = int(embedding_dim)
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes))
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        print('Initialised AM-Softmax m=%.3f s=%.3f' % (self.m, self.s))
        print(
            'Embedding dim is {}, number of speakers is {}'.format(
                embedding_dim, num_classes
            )
        )

    def forward(self, x, label=None, flagSyn=False):
        if flagSyn:
            raise ValueError("Plain AM-Softmax does not define synthetic classes")
        if label is None:
            raise ValueError("AM-Softmax requires speaker labels")
        assert x.size(0) == label.size(0)
        assert x.size(1) == self.in_feats

        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        w_norm = torch.nn.functional.normalize(self.W, p=2, dim=0)
        cosine = torch.mm(x_norm, w_norm)
        margin = torch.zeros_like(cosine)
        margin.scatter_(1, label.view(-1, 1), self.m)
        logits = self.s * (cosine - margin)
        loss = self.ce(logits, label)
        acc = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        return loss, acc, None
