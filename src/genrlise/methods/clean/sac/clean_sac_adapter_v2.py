import copy

from genrlise.methods.clean.sac.clean_sac_adapter_v1 import CleanSACAdapterV1
from genrlise.methods.clean.sac.networks.adapter import ActorAdapterOnlyFeatures


class CleanSACAdapterV2(CleanSACAdapterV1):
    """
        Only uses features.
    """

        
    def _get_actor_class(self):
        return ActorAdapterOnlyFeatures
    
    def _get_critic_kwargs(self, kwargs): 
        if self.exp_conf("method/args/do_ignore_args_for_critic", False):
            kwargs['adapter_kwargs'] = copy.deepcopy(kwargs['adapter_kwargs'])
            kwargs['adapter_kwargs'] = {k: v for k, v in kwargs['adapter_kwargs'].items() if k in ['skip_connection', 'context_size', 'hnet_kwargs']}
            kwargs['adapter_kwargs']['hnet_kwargs'] = {k: v for k, v in kwargs['adapter_kwargs']['hnet_kwargs'].items() if k in ['hypernetwork_layers']}
        return kwargs