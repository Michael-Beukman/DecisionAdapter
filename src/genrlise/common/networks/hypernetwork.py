from typing import Any, Dict, List, Tuple, Union
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
from hypnettorch.mnets.mlp import MLP
from hypnettorch.mnets.mnet_interface import MainNetInterface
from torch import nn
import torch
from blitz.modules import BayesianLinear
from genrlise.common.networks.main_network import MainNetwork
from genrlise.common.networks.my_clean_mlp_hnet import MyCleanHMLP
import numpy as np
from genrlise.common.networks.network_initialisation import get_weight_initialisation_func
from genrlise.common.networks.weight_generator import WeightGenerator

def _get_hidden_size(
    num_weights_mnet, alpha, ctx_size, inp_embeddings, return_size=False
):
    def getabc(good_h):
        num_chunks = np.ceil(num_weights_mnet / alpha / good_h)  # upper bound
        a = 1 + alpha
        b = (ctx_size + 1) + 1 + alpha + inp_embeddings
        c = -num_weights_mnet + inp_embeddings * num_chunks
        return a, b, c

    def _count_params(good_h):
        a, b, c = getabc(good_h)
        return a * good_h ** 2 + b * good_h + (c + num_weights_mnet)

    good_h = 1
    while 1:
        a, b, c = getabc(good_h)

        def roots():
            f1 = np.sqrt(b ** 2 - 4 * a * c)
            return int((-b - f1) / (2 * a)), int((-b + f1) / (2 * a))

        rs = roots()
        new_good_h = rs[-1]
        if new_good_h <= good_h or _count_params(new_good_h) > num_weights_mnet:
            break
        good_h = new_good_h
    if return_size:
        return alpha * good_h, [good_h, good_h], _count_params(good_h)
    return alpha * good_h, [good_h, good_h]

class Hypernetwork(nn.Module):
    """
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Tuple[int],
        activation_fn=torch.nn.ReLU(),
        context_size: int = 2,
        hypernetwork_layers: List[int] = [100, 100],
        norm_output: bool = False,
        output_func: str = None,
        verbose: bool = True,
        norm_output_per_layer: bool = False,
        do_principled_init: bool = False,
        input_func: str = None,
        return_weights_on_forward: bool = False,
        log_weight_norms: bool = True,
        chunk_alpha=10,
        chunk_embedding_size=8,
        hypernetwork_activation_function: str = "relu",
        hypernetwork_initialisation: str = None,
        use_unconditional_inputs: bool = False,
        use_cond_embeddings: bool = False,
        threshold_to_assert_sizes='default',
        use_context_feature_extractor: str=None, # either None, "bayesian" or "sigmoid"
        has_activation_func_at_end: bool = False,
        override_hypernet: bool = False,          # if true, the net does not output anything, instead 
        use_joint_hypernet: bool = False,
        context_feature_extractor_dimension: int = None
    ) -> None:
        """This is the main hypernetwork class. This effectively generates neural network weights given some context.
           See 
                Ha et al.; HyperNetworks; Arxiv; 2016
                Oswald et al.; Continual learning with hypernetworks

        Args:
            in_dim (int): The input dimension of the main network
            out_dim (int): The output dimension of the main network
            layers (Tuple[int]): The layers of the main network, given as a list/tuple of integers
            activation_fn (_type_, optional): Activation function to use in the main network. Defaults to torch.nn.ReLU().
            context_size (int, optional): The dimension of the context. Defaults to 2.
            hypernetwork_layers (List[int], optional): The hidden layers of the hypernetwork. Defaults to [100, 100].
            norm_output (bool, optional): Should the weights be normalised to have a norm of 1. Defaults to False.
            output_func (str, optional): The output function to be performed on the weights, ['tanh', 'log', 'symlog', 'selu']. Defaults to None.
            verbose (bool, optional): Should information be printed. Defaults to True.
            norm_output_per_layer (bool, optional): _description_. Defaults to False.
            do_principled_init (bool, optional): _description_. Defaults to False.
            input_func (str, optional): _description_. Defaults to None.
            return_weights_on_forward (bool, optional): _description_. Defaults to False.
            log_weight_norms (bool, optional): _description_. Defaults to True.
            chunk_alpha (int, optional): _description_. Defaults to 10.
            chunk_embedding_size (int, optional): _description_. Defaults to 8.
            hypernetwork_activation_function (str, optional): _description_. Defaults to "relu".
            hypernetwork_initialisation (str, optional): _description_. Defaults to None.
            use_unconditional_inputs (bool, optional): _description_. Defaults to False.
            use_cond_embeddings (bool, optional): _description_. Defaults to False.
            threshold_to_assert_sizes (str, optional): _description_. Defaults to 'default'.
            use_context_feature_extractor (str, optional): _description_. Defaults to None.
            override_hypernet (bool, optional): _description_. Defaults to False.
            insteaduse_joint_hypernet (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self._base_model: torch.Tensor = None
        self._mean_model: torch.Tensor = None
        self._std_model: torch.Tensor = None
        self._min_model: torch.Tensor = None
        self._max_model: torch.Tensor = None
        
        self.override_hypernet: bool = override_hypernet

        self.weight_generator_override: WeightGenerator = None
        
        self.hypernetwork_initialisation = hypernetwork_initialisation
        self.do_principled_init = do_principled_init
        self.use_unconditional_inputs = use_unconditional_inputs
        self.threshold_to_assert_sizes = threshold_to_assert_sizes
        self.use_context_feature_extractor = use_context_feature_extractor
        self.context_feature_extractor = None
        if self.use_context_feature_extractor is not None:
            print("GOT feature extactor arg", use_context_feature_extractor, context_feature_extractor_dimension)
            if context_feature_extractor_dimension is None:
                context_feature_extractor_dimension = context_size
            if self.use_context_feature_extractor == 'bayesian':
                print('using bayesian context feature extractor')
                self.context_feature_extractor = nn.Sequential(
                    BayesianLinear(context_size, 32),
                    nn.ReLU(),
                    BayesianLinear(32, context_feature_extractor_dimension, ),
                )
            elif self.use_context_feature_extractor == 'sigmoid':
                print('using sigmoid context feature extractor')
                self.context_feature_extractor = nn.Sequential(
                    nn.Linear(context_size, 32),
                    nn.Sigmoid(),
                    nn.Linear(32, context_feature_extractor_dimension, ),
                    nn.Sigmoid(),
                )
            elif self.use_context_feature_extractor == 'relu':
                print('using relu context feature extractor')
                self.context_feature_extractor = nn.Sequential(
                    nn.Linear(context_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, context_feature_extractor_dimension, ),
                )
            elif self.use_context_feature_extractor == 'full_relu':
                print('using full relu context feature extractor')
                self.context_feature_extractor = nn.Sequential(
                    nn.Linear(context_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, context_feature_extractor_dimension, ),
                    nn.ReLU(),
                )
            else:
                assert False, f"Invalid value for use_context_feature_extractor: '{use_context_feature_extractor}'"
            context_size = context_feature_extractor_dimension
        if self.do_principled_init:
            assert (
                self.hypernetwork_initialisation is None
            ), "Cannot have both principled and hypernetwork_initialisation"
        self.has_activation_func_at_end = has_activation_func_at_end
        old_layers = layers
        self.mnet = MainNetwork(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=layers,
            activation_fn=activation_fn,
            has_activation_func_at_end=has_activation_func_at_end
        )
        linear_shapes = MLP.weight_shapes(
            n_in=self.mnet.in_dim,
            n_out=self.mnet.out_dim,
            hidden_layers=self.mnet.hidden_layers,
            use_bias=True,
        )
        self.output_func = output_func
        self.norm_output_per_layer = norm_output_per_layer

        self.hypernetwork_activation_function = hypernetwork_activation_function
        hnet_activation_function = {"relu": torch.nn.ReLU, "selu": torch.nn.SELU}[hypernetwork_activation_function.lower()]()
        if hypernetwork_layers == "less":
            hypernetwork_layers = 1.0
        self.number_of_weights_to_generate = MainNetInterface.shapes_to_num_weights(linear_shapes)
        if type(hypernetwork_layers) == float:
            # This sets up a chunked MLP with the appropriate size.
            counts = MainNetInterface.shapes_to_num_weights(linear_shapes)

            chunk_size, layers, _total = _get_hidden_size(
                counts * hypernetwork_layers,
                chunk_alpha,
                context_size,
                chunk_embedding_size,
                return_size=True,
            )
            print(f"Generating HNet with {hypernetwork_layers=}, {chunk_size=}, {_total=} {layers=}")
            if layers == [1, 1] and _total >= counts * hypernetwork_layers:
                layers = [1]
            args_to_hnet = dict(
                target_shapes=linear_shapes,
                chunk_size=chunk_size,
                cond_in_size=context_size,
                layers=tuple(layers),
                num_cond_embs=0,
                no_cond_weights=True,
                chunk_emb_size=chunk_embedding_size,
                activation_fn=hnet_activation_function,
                verbose=verbose,
            )
            
            if use_cond_embeddings:
                assert self.use_unconditional_inputs == False, "Cannot have both use_cond_embeddings and use_unconditional_inputs True at once"
                args_to_hnet['num_cond_embs'] = 8
                args_to_hnet['no_cond_weights'] = False

            if self.use_unconditional_inputs:
                args_to_hnet['uncond_in_size'] = context_size
                args_to_hnet['cond_in_size'] = 0
                args_to_hnet['no_uncond_weights'] = False
            
            self.hnet = ChunkedHMLP(**args_to_hnet)
            if do_principled_init:
                self.hnet.apply_chunked_hyperfan_init()
            def _do_assert(t=1):
                assert (
                    self.hnet.num_params <= t * counts
                ), f"Hnet params ({self.hnet.num_params}) must be less than the Mnet Params ({t} * {counts}). {in_dim} -> {old_layers} -> {out_dim}. Hnet = {chunk_size}, {layers}"
            if self.threshold_to_assert_sizes == 'default':
                _do_assert()
            else: 
                _do_assert(self.threshold_to_assert_sizes)

        else:
            if use_unconditional_inputs:
                assert False, "Cannot use use_unconditional_inputs with a non-chunked hypernet"
            self.hnet = MyCleanHMLP(
                linear_shapes,
                cond_in_size=context_size,
                layers=tuple(hypernetwork_layers),
                activation_fn=hnet_activation_function,
                verbose=verbose,
            )
            if do_principled_init:
                self.hnet.apply_hyperfan_init()

        if self.hypernetwork_initialisation is not None:
            if self.hypernetwork_initialisation == "selu":
                for w in self.hnet.layer_weight_tensors:
                    torch.nn.init.kaiming_normal_(w, nonlinearity="linear")

        self.out_dim = out_dim
        self.norm_output = norm_output
        self.input_func = input_func
        L = 0
        self.return_weights_on_forward = return_weights_on_forward
        self.log_weight_norms = log_weight_norms
        self.total_weight_norm: float = 0
        self.context_size = context_size
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        return_generated_weights: Union[bool, int] = None,
    ):
        """Performs a forward pass

        Args:
            x (torch.Tensor): _description_
            context (torch.Tensor): _description_
            return_generated_weights (bool, optional): If 100, returns just the weights. If true, returns both. Defaults to None.

        Returns:
            _type_: _description_
        """
        batch_size = x.shape[0]
        context = context.reshape(batch_size, -1)
        if self.input_func is not None:
            if self.input_func.lower() == "div_100":
                context = context / 100
            elif self.input_func.lower() == "log":

                A = (context > 0).int().float()
                B = (context < 0).int().float()
                context = (1 - A) * context + A * torch.log(context * A + 1)
                context = (1 - B) * context + B * -torch.log(-context * B + 1)
                assert not torch.any(torch.isnan(context))
            else:
                raise Exception("Unknown input function", self.input_func)
        if self.context_feature_extractor is not None:
            context = self.context_feature_extractor(context)

        if self.override_hypernet:
            weights = self.weight_generator_override.get_weights(context)
        elif self.use_unconditional_inputs:
            weights = self.hnet.forward(uncond_input=context, ret_format="flattened")
        else:
            weights = self.hnet.forward(cond_input=context, ret_format="flattened")
        
        if self.norm_output:
            norms = torch.linalg.norm(weights, dim=-1, keepdim=True)
            assert norms.shape == (batch_size, 1)
            weights = weights / norms
            
        
        if self.output_func is not None:
            if self.output_func.lower() == "tanh":
                weights = torch.tanh(weights)
            elif self.output_func.lower() == "log":
                weights = torch.log(torch.relu(weights) + 1e-5)
                pass
            elif self.output_func.lower() == "symlog":
                old_shape = weights.shape
                A = (weights > 0).int().float()
                B = (weights < 0).int().float()
                weights = (1 - A) * weights + A * torch.log(weights * A + 1)
                weights = (1 - B) * weights + B * -torch.log(-weights * B + 1)
                assert not torch.any(torch.isnan(weights))
                weights = weights.reshape(old_shape)
            elif self.output_func.lower() == "selu":
                weights = torch.selu(weights)
            else:
                raise Exception("Unknown activation function", self.output_func)
        
        if self._mean_model is not None:
            weights = weights * self._std_model + self._mean_model
        
        if self._min_model is not None:
            weights = weights * (self._max_model - self._min_model) + self._min_model
            
        if self._base_model is not None:
            weights = weights + self._base_model

        if return_generated_weights == 100: return weights
        output = self.mnet.forward(x, weights)
        if (
            (return_generated_weights is None) and self.return_weights_on_forward
        ) or return_generated_weights:
            return output, weights
        return output

    def set_base_model(self, base_model: torch.Tensor):
        # Sets the base model for this network. This will be added to any prediction weights
        print("BASE", base_model)
        self._base_model = (base_model)
        
    def set_mean_std(self, mean: torch.Tensor, std: torch.Tensor):
        assert self._min_model is None
        self._mean_model = (mean)
        self._std_model = (std)
        
    def set_min_max(self, min: torch.Tensor, max: torch.Tensor):
        assert self._mean_model is None
        self._min_model = (min)
        self._max_model = (max)

    def set_weight_generator(self, model: WeightGenerator):
        self.weight_generator_override = model
    
    
    def params(self):
        return self.hnet.internal_params

    def do_initialisation(self, init: Dict[str, Any]):
        print("Hey, hnet here is doing inits: ", init)
        func = get_weight_initialisation_func(init)
        self.hnet.apply(func)