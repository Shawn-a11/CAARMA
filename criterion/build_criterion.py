from criterion.amsoftmax_mix_gan import amsoftmax_gan

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_spk'],
            m=config.get('am_margin', 0.2),
            s=config.get('am_scale', 30),
        )
    else:
        raise NotImplementedError

    return criterion
