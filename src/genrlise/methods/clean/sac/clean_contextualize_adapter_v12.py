
import torch
from genrlise.methods.clean.sac.base_clean_sac import train_sac
from genrlise.methods.clean.sac.clean_sac_adapter_v1 import CleanSACAdapterV1
from genrlise.methods.clean.sac.networks.good_adapters import AdapterActorContextualize, AdapterCriticContextualize

K = 205
class CleanSACContextualizeAdapterV12(CleanSACAdapterV1):
    """
        Benjamins et al. 2022, cGate
    """

    def _get_actor_class(self):
        return AdapterActorContextualize

    def _get_critic_class(self):
        return AdapterCriticContextualize
    
    def _get_learn_function(self):  
        return train_sac

    
    def _get_actor_kwargs(self, kwargs): 
        kwargs['net_arch']                  = self.exp_conf("method/args/net_arch", [K, K])
        kwargs['adapter_net_arch']          = self.exp_conf("method/args/adapter_net_arch", [K])
        kwargs['final_policy_arch']         = self.exp_conf("method/args/adapter_actor_action_arch", [K])
        kwargs['final_log_std_arch']         = self.exp_conf("method/args/adapter_actor_log_std_arch", None)
        
        
        kwargs['trunk_has_act_end']         = self.exp_conf("method/args/trunk_has_act_end", False)
        kwargs['adapter_has_act_end']         = self.exp_conf("method/args/adapter_has_act_end", False)
        if 'adapter_kwargs' in kwargs: del kwargs['adapter_kwargs']
        return kwargs
    
    def _get_critic_kwargs(self, kwargs): 
        base = {k: v for k, v in kwargs.items() if k == 'net_arch' or k == 'adapter_net_arch' or k == 'context_dim'}
        if 'adapter_kwargs' in kwargs: del kwargs['adapter_kwargs']
        base['net_arch']                  = self.exp_conf("method/args/net_arch", [K, K])
        base['adapter_net_arch']          = self.exp_conf("method/args/adapter_net_arch", [K])
        base['final_arch']         = self.exp_conf("method/args/adapter_critic_final_arch", [K])
        return base
    
    def _get_learn_kwargs(self):
        D = {}
        D['should_train_critic'] = self.exp_conf("method/args/should_train_critic", True)
        return D
    
    def load_method_from_file(self, filename: str):
        self.model = torch.load(filename, map_location=self.device)