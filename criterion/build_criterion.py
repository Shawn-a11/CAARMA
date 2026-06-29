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
                                  mixup_constraint=config.get('mixup_constraint', 'none'),
                                  train_csv=config.get('dataset'),
                                  meta_csv=config.get('meta_csv'),
                                  pair_filter=config.get('pair_filter', 'none'),
                                  cosine_pct_low=config.get('cosine_pct_low', 0.60),
                                  cosine_pct_high=config.get('cosine_pct_high', 0.90),
                                  skip_invalid_pairs=config.get('skip_invalid_pairs', False))
    elif config['criterion'] == 'AMSoftmax':
        criterion = amsoftmax(embedding_dim=config['embedding_dim'], num_classes=config['num_spk'], m=0.2, s=30)
    else:
        raise NotImplementedError

    return criterion
