from typing import List
import torch.nn as nn

def get_feedforward_model(INPUT_FEATURES: int, OUTPUT_FEATURES: int, layers: List[int] = [256, 256],HAS_BIAS: bool = True, act = nn.ReLU, has_act_final: bool = False) -> nn.Module:
    """Common utility to get a feedforward model, given a few inputs

    Args:
        INPUT_FEATURES (int): Number of input nodes
        OUTPUT_FEATURES (int): Number of outputs nodes
        layers (List[int], optional): List of layer sizes, [] maps input directly to output. Defaults to [256, 256].
        HAS_BIAS (bool, optional): . Defaults to True.
        act (_type_, optional): Activation function. Defaults to nn.ReLU.
        has_act_final (bool, optional): Has an activation function at the end. Defaults to False.

    Returns:
        nn.Module: 
    """
    
    all_vals = [INPUT_FEATURES] + layers + [OUTPUT_FEATURES]
    
    all_layers = []
    for i in range(len(all_vals) - 1):
        all_layers.append(
            nn.Linear(in_features=all_vals[i], out_features=all_vals[i + 1], bias=HAS_BIAS)
        )
        if i != len(all_vals) - 2 or has_act_final:
            all_layers.append(act())
    
    return nn.Sequential(*all_layers)

if __name__ == '__main__':
    print(get_feedforward_model(10, 1, [256, 256]))