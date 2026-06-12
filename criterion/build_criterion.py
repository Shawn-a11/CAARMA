from criterion.amsoftmax_mix_gan import amsoftmax_gan
from criterion.amsoftmax import amsoftmax

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_spk'],
            m=0.2, s=30,
            # attribute-constrained mixup (gender / nationality / none)
            mixup_constraint=config.get('mixup_constraint', 'none'),
            train_csv=config.get('dataset'),
            meta_csv=config.get('meta_csv'),
        )
    elif config['criterion'] == 'AMSoftmax':
        criterion = amsoftmax(embedding_dim=config['embedding_dim'], num_classes=config['num_spk'], m=0.2, s=30)
    else:
        raise NotImplementedError

    return criterion
