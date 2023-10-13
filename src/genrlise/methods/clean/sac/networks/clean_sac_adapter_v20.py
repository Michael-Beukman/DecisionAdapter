
from genrlise.methods.clean.sac.clean_sac_adapter_v1 import CleanSACAdapterV1
from genrlise.methods.clean.sac.networks.adapter import SoftQNetworkAdapterTrunkNoActivation


class CleanSACAdapterV20(CleanSACAdapterV1):
    """
        Critic trunk and no act
    """

    def _get_critic_class(self):
        return SoftQNetworkAdapterTrunkNoActivation