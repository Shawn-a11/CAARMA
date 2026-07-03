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
                                  synth_per_batch=config.get('synth_per_batch', 0),
                                  boundary_utility=config.get('boundary_utility', False),
                                  boundary_candidate_pool=config.get('boundary_candidate_pool', 64),
                                  boundary_tau_parent=config.get('boundary_tau_parent', 0.95),
                                  boundary_tau_low=config.get('boundary_tau_low', 0.0),
                                  boundary_tau_high=config.get('boundary_tau_high', 0.90),
                                  boundary_margin_gap=config.get('boundary_margin_gap', 0.05),
                                  crp_count_cap=config.get('crp_count_cap', 20.0),
                                  crp_count_decay=config.get('crp_count_decay', 0.02),
                                  crp_utility_kappa=config.get('crp_utility_kappa', 2.0),
                                  utility_ema_beta=config.get('utility_ema_beta', 0.9))
    elif config['criterion'] == 'AMSoftmax':
        criterion = amsoftmax(embedding_dim=config['embedding_dim'], num_classes=config['num_spk'], m=0.2, s=30)
    else:
        raise NotImplementedError

    return criterion
