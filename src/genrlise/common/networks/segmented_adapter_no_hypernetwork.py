from turtle import forward
from common.utils import load_compressed_pickle, save_compressed_pickle
from genrlise.common.imports import *
from torch import nn
import torch
import numpy as np

from genrlise.common.networks.hypernetwork import Hypernetwork
from genrlise.common.networks.network_initialisation import get_weight_initialisation_func
from genrlise.methods.clean.sac.networks.common import get_feedforward_model

def swap_if_none(a, b):
    if a is None: return b
    return a

class SegmentedAdapterNoHyperNetwork(nn.Module):
    """ Copied from the segmented hypernetwork class, just does not use a hypernetwork; instead the adapter takes in the concatenation of the state and context
    """

    def __init__(self,  in_dim: int,
                        out_dim: int,
                        layers: Tuple[int],
                        activation_fn=torch.nn.ReLU(),
                        context_size: int = 2,
                        should_adapt_ignore_layers: bool = True,
                        has_activation_func_at_end: bool = False,
                        hnet_kwargs: Dict[str, Any] = {},
                        skip_connection: bool = False,
                        adapter_layers: List[int] = [32],
                        adapter_activation_fn_at_end: bool = True,
                        adapter_override_output_dim: Union[int, None] = None,
                        has_final_main_network: bool = True,
                        is_adapter_concat: bool = False,
                        
                        use_context_feature_extractor: str=None, # either None, "bayesian" or "sigmoid"
                        context_feature_extractor_dimension: int = None,
                        
                        put_adapter_at_start: bool                              = False,
                        put_adapter_after_first_layer: bool                     = False,
                        put_adapter_before_last_layer: bool                     = True,
                        put_adapter_at_end: bool                                = False,
                        
                        shallow_override_put_adapter_at_start: bool             = None,
                        shallow_override_put_adapter_after_first_layer: bool    = None,
                        shallow_override_put_adapter_before_last_layer: bool    = None,
                        shallow_override_put_adapter_at_end: bool               = None
                        ) -> None:
        """This creates the adapter model

        Args:
            in_dim (int): The dimension of the problem, i.e. the number of state variables.
            out_dim (int): The dimension of the output of the network
            layers (Tuple[int]): The number of layers in the main network
            activation_fn (_type_, optional): The activation function of the main network. Defaults to torch.nn.ReLU().
            context_size (int, optional): The context dimension. Defaults to 2.
            should_adapt_ignore_layers (bool, optional): This should be true now, the adapter just adds to the layers of the main net, it does not replace any. Defaults to True.
            has_activation_func_at_end (bool, optional): This is true if the main network's output must be processed by an activation function. Defaults to False.
            hnet_kwargs (Dict[str, Any], optional): These are the kwargs to be given to the hypernetwork. Defaults to {}.
            skip_connection (bool, optional): If this is true, there is a skip connection that makes it possible to bypass the adapter entirely. Defaults to False.

            adapter_layers (List[int], optional): The layers of the adapter network. Defaults to [32],
            adapter_activation_fn_at_end (bool, optional): If true, does the adapter have an activation function at the end .Defaults to True,
            adapter_override_output_dim (Union[int, None], optional): Override the adapter's output dimension, instead of the normal value. Defaults to= None,
            has_final_main_network (bool, optional): If False, the adapter is the final layer. Defaults to True
            is_adapter_concat (bool, optional): If true, the main trunk of the model also takes in the context. Defaults to False

            put_adapter_at_start (bool, optional): Whether or not to put the adapter at the very start of the network
            put_adapter_after_first_layer (bool, optional): Whether or not to put the adapter after the first layer
            put_adapter_before_last_layer (bool, optional): Whether or not to put the adapter before the last layer. The last layer might not exist.
            put_adapter_end (bool, optional): Whether or not to put the adapter at the very end
        """
        super().__init__()
        # so, if should_adapt_ignore_layers == False
        # main_network_initial
            # (in_dim, layers[0]) -> (layers[0], layers[1]) -> ... -> (layers[-3], layers[-2])
        
        # hypernetwork_adapter
            # (layers[-2], layers[-1])
        
        # self.main_network_final
            # (layers[-1], out_dim)
            
        # else
        # main_network_initial
            # (in_dim, layers[0]) -> (layers[0], layers[1]) -> ... -> (layers[-2], layers[-1])
        
        # hypernetwork_adapter
            # (layers[-1], layers[-1])
        
        # self.main_network_final
            # (layers[-1], out_dim)
        print(f"Adapter Args  {put_adapter_at_start=} {put_adapter_after_first_layer=} {put_adapter_before_last_layer=} {put_adapter_at_end=} {skip_connection=} {hnet_kwargs=}", )
        assert should_adapt_ignore_layers, "For now, only this mode is supported"
        assert is_adapter_concat == False, "For now, only this mode is supported"
        assert adapter_override_output_dim is None, "For now, only this mode is supported"
        self.is_adapter_concat = is_adapter_concat
        self.skip_connection = skip_connection
        self.has_final_main_network = has_final_main_network
        self.has_activation_func_at_end = has_activation_func_at_end
        self.activation_fn = activation_fn
        self.out_dim = out_dim
        
        all_layers_to_use = [in_dim] + list(layers) + [out_dim]
        
        print("Skip connection: ", self.skip_connection)
        print("adapter here received use_context_feature_extractor and context_feature_extractor_dimension", use_context_feature_extractor, context_feature_extractor_dimension)
        print("adapter here using layers", all_layers_to_use)
        
        self.adapters = [None for i in range(min(4, len(all_layers_to_use)))]
        def get_hnet(in_dim, out_dim, should_use_activation_fn_at_end=adapter_activation_fn_at_end):
            temp_act = activation_fn
            # print("MIKE", temp_act, temp_act == nn.ReLU(), isinstance(temp_act, nn.ReLU))
            if isinstance(temp_act, nn.ReLU):
                temp_act = nn.ReLU
            return get_feedforward_model(in_dim + context_size, out_dim, layers=adapter_layers, act=temp_act or nn.ReLU, has_act_final=should_use_activation_fn_at_end)

        if len(all_layers_to_use) == 2:
            # Input maps directly to output, no hidden layers
            self.main_nets = [
                nn.Sequential(*([nn.Linear(all_layers_to_use[0], all_layers_to_use[1])] + ([activation_fn] if has_activation_func_at_end else [])))
            ]
            
            shallow_override_put_adapter_at_start = swap_if_none(shallow_override_put_adapter_at_start, put_adapter_at_start)
            shallow_override_put_adapter_at_end = swap_if_none(shallow_override_put_adapter_at_end, put_adapter_at_end)
            shallow_override_put_adapter_before_last_layer = swap_if_none(shallow_override_put_adapter_before_last_layer, put_adapter_before_last_layer)
            shallow_override_put_adapter_after_first_layer = swap_if_none(shallow_override_put_adapter_after_first_layer, put_adapter_after_first_layer)
            
            if shallow_override_put_adapter_at_start or shallow_override_put_adapter_before_last_layer:
                self.adapters[0] = get_hnet(in_dim=all_layers_to_use[0], out_dim=all_layers_to_use[0])
            if shallow_override_put_adapter_at_end or shallow_override_put_adapter_after_first_layer:
                self.adapters[1] = get_hnet(in_dim=all_layers_to_use[1], out_dim=all_layers_to_use[1], should_use_activation_fn_at_end=has_activation_func_at_end)
            
        elif len(all_layers_to_use) == 3:
            # Input maps to F, F maps to output; one hidden layer
            self.main_nets = [
                nn.Sequential(*([nn.Linear(all_layers_to_use[0], all_layers_to_use[1])] + [activation_fn])),
                nn.Sequential(*([nn.Linear(all_layers_to_use[1], all_layers_to_use[2])] + ([activation_fn] if has_activation_func_at_end else []))),
            ]
            
            if put_adapter_at_start:
                self.adapters[0] = get_hnet(in_dim=all_layers_to_use[0], out_dim=all_layers_to_use[0])
            
            if put_adapter_after_first_layer or put_adapter_before_last_layer:
                self.adapters[1] = get_hnet(in_dim=all_layers_to_use[1], out_dim=all_layers_to_use[1])
                                
            if put_adapter_at_end:
                self.adapters[2] = get_hnet(in_dim=all_layers_to_use[2], out_dim=all_layers_to_use[2], should_use_activation_fn_at_end=has_activation_func_at_end)
            
        elif len(all_layers_to_use) >= 4:
            # Input -> F1; F1 -> F2; ... Fn -> Out
            starts = nn.Sequential(*([nn.Linear(all_layers_to_use[0], all_layers_to_use[1])] + [activation_fn]))
            ends = nn.Sequential(*([nn.Linear(all_layers_to_use[-2], all_layers_to_use[-1])] + ([activation_fn] if has_activation_func_at_end else [])))
            tmp = all_layers_to_use[1:-1]
            to_add = []
            for l in zip(tmp, tmp[1:]):
                to_add.append(nn.Linear(l[0], l[1]))
                to_add.append(activation_fn)
            self.main_nets = [
                starts,
                nn.Sequential(*to_add),
                ends,
            ]
                    
            if put_adapter_at_start:
                self.adapters[0] = get_hnet(in_dim=all_layers_to_use[0], out_dim=all_layers_to_use[0])
            
            if put_adapter_after_first_layer:
                self.adapters[1] = get_hnet(in_dim=all_layers_to_use[1], out_dim=all_layers_to_use[1])
            
            if put_adapter_before_last_layer:
                self.adapters[2] = get_hnet(in_dim=all_layers_to_use[-2], out_dim=all_layers_to_use[-2])
                                
            if put_adapter_at_end:
                self.adapters[3] = get_hnet(in_dim=all_layers_to_use[-1], out_dim=all_layers_to_use[-1], should_use_activation_fn_at_end=has_activation_func_at_end)
        print("LENS", len(self.adapters), len(self.main_nets), len(all_layers_to_use), self.adapters)
        self.adapters = nn.ModuleList(self.adapters)
        self.main_nets = nn.ModuleList(self.main_nets)
        
        self.is_adapter_enabled = True
    
    def enable_disable_adapter(self, enabled: bool):
        self.is_adapter_enabled = enabled
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, return_generated_weights: bool = False):
        # print("Hey there. Context=", context)
        assert return_generated_weights == False
        assert len(self.adapters) == len(self.main_nets) + 1
        features = x
        for i in range(len(self.main_nets)):
            if self.adapters[i] is not None and self.is_adapter_enabled:
                combined = torch.cat([features, context], dim=1)
                adapted = self.adapters[i](combined) # Here, in this class, we just concat the context and state at the adapter module
                # print("Adapting", features.shape, 'to', adapted.shape)
                if self.skip_connection:
                    # print("Adding skip connection")
                    assert adapted.shape == features.shape
                    adapted = adapted + features # skip connection.
                features = adapted
            features = self.main_nets[i](features)
            # print("Main Net outputted", features.shape)
        if self.adapters[-1] is not None and self.is_adapter_enabled:
            # print("Final adapter")
            adapted = self.adapters[-1](x=features, context=context)
            if self.skip_connection:
                assert adapted.shape == features.shape
                adapted = adapted + features # skip connection.
            features = adapted
        return features

    def freeze_adapter(self, unfreeze = False):
        assert False, "DO not call"
        for param in self.hypernetwork_adapter.parameters():
            param.requires_grad = unfreeze
            
    def freeze_main_net(self, unfreeze = False):
        assert False, "DO not call"
        self.freeze_main_net_before(unfreeze)
        self.freeze_main_net_after(unfreeze)
    
    def freeze_main_net_before(self, unfreeze=False):
        assert False, "DO not call"
        for param in self.main_network_initial.parameters():
            param.requires_grad = unfreeze
    
    def freeze_main_net_after(self, unfreeze=False):
        assert False, "DO not call"
        for param in self.main_network_final.parameters():
            param.requires_grad = unfreeze
            
    def do_initialisation(self, init: Dict[str, Any]):
        method = init.get("method", 'default')
        if method == "beck_normc":
            print("USING BECK NORMC")
            std = init.get("std", 1)
            for adapt in self.adapters:
                if adapt is None: continue
            
                torch.nn.init.zeros_(adapt.hnet._hnet._layer_weight_tensors[-1])
                t = torch.randn(adapt.hnet._hnet._layer_bias_vectors[-1].shape, device=adapt.hnet._hnet._layer_bias_vectors[-1].device)
                t *= std / torch.sqrt((t**2).sum())
                with torch.no_grad():
                    adapt.hnet._hnet._layer_bias_vectors[-1].copy_(t)
            return
        func = get_weight_initialisation_func(init)
        print("Hey adapter here is doing init", init, func)
        for m in self.main_nets:
            m.apply(func)
        for a in self.adapters:
            if a is not None: a.apply(func)