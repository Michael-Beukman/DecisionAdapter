
from genrlise.methods.clean.sac.clean_sac_adapter_v1 import CleanSACAdapterV1
from genrlise.methods.clean.sac.networks.adapter import ActorAdapterOnlyMeanNotHypernetwork, SoftQNetworkAdapterNotHypernetwork


class CleanSACAdapterV1NoHypernetwork(CleanSACAdapterV1):
        
    def _get_actor_class(self):
        return ActorAdapterOnlyMeanNotHypernetwork
    
    def _get_critic_class(self):
        return SoftQNetworkAdapterNotHypernetwork
