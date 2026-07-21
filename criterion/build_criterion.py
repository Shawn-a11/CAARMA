from criterion.amsoftmax_mix_gan import amsoftmax_gan
from criterion.amsoftmax import amsoftmax

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(embedding_dim=config['embedding_dim'], num_classes=config['num_spk'], m=0.2, s=30,
                                  persistence=config.get('persistence', False),
                                  slerp_t=config.get('slerp_t', 0.5),
                                  synth_bank_size=config.get('synth_bank_size', 10),
                                  synth_max_factor=config.get('synth_max_factor', 4),
                                  pair_strategy=config.get('pair_strategy', 'fixed_nn'),
                                  crp_alpha=config.get('crp_alpha', 1.0),
                                  crp_topk=config.get('crp_topk', 4),
                                  reuse_policy=config.get('reuse_policy', 'popularity'))
    elif config['criterion'] == 'AMSoftmax':
        criterion = amsoftmax(embedding_dim=config['embedding_dim'], num_classes=config['num_spk'], m=0.2, s=30)
    else:
        raise NotImplementedError

    return criterion
