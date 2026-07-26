from criterion.amsoftmax_mix_gan import amsoftmax_gan
from criterion.amsoftmax import amsoftmax

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_spk'],
            margin=float(config.get('am_margin', 0.2)),
            scale=float(config.get('am_scale', 30.0)),
        )
    elif config['criterion'] == 'AMSoftmax':
        criterion = amsoftmax(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_spk'],
            margin=float(config.get('am_margin', 0.2)),
            scale=float(config.get('am_scale', 30.0)),
        )
    else:
        raise NotImplementedError

    return criterion
