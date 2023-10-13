import torch
import torch.nn as nn

class WeightGenerator(nn.Module):
    """NOTE: This is not really used.
    
    
    This is a way to override the hypernetwork, to generate weights in a different way, but still take advantage of how integrated the hnet is with stable baselines, etc.
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def get_weights(self, context: torch.Tensor) -> torch.Tensor:
        """Take in a context (use or ignore it) and generate a set of weights

        Args:
            context (torch.Tensor): 

        Returns:
            torch.Tensor: The set of weights
        """
        raise NotImplementedError()