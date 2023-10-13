from functools import partial
from typing import Any, Dict
import torch
def weights_init_small(std, m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        torch.nn.init.normal_(m.weight, mean=0, std=std)
        m.bias.data.fill_(0)

def weights_default(m):
    return


def get_weight_initialisation_func(init: Dict[str, Any]):
    method = init.get("method", 'default')
    if method  == 'default': return weights_default
    elif method == 'small':
        return partial(weights_init_small, init.get("std"))
    else:
        raise Exception(f"Invalid option {method}")