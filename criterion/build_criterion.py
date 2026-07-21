from criterion.amsoftmax_mix_gan import amsoftmax_gan
from criterion.amsoftmax import amsoftmax

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_spk'],
            margin=0.2,
            scale=30,
            prototype_only_virtual=config.get('prototype_only_virtual', False),
            virtual_negative_topk=config.get('virtual_negative_topk', 4),
            virtual_negative_t=config.get('virtual_negative_t', 0.5),
            virtual_negatives_per_batch=config.get('virtual_negatives_per_batch', 8),
        )
    elif config['criterion'] == 'AMSoftmax':
        criterion = amsoftmax(embedding_dim=config['embedding_dim'], num_classes=config['num_spk'], m=0.2, s=30)
    else:
        raise NotImplementedError

    return criterion
