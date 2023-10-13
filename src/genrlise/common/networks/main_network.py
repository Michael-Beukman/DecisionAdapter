from typing import List, Tuple
import torch

class MainNetwork:
    """
        This is a MainNetwork, pretty much a linear neural network that is given the weights to be used when predicting at runtime
        I.e., this has no parameters to learn.
        
        This is useful for two main cases:
        - When a hypernetwork is used, the MainNetwork can perform a forward pass using the generated weights
        - Likewise, when using a Bayesian Neural Network, the sampled weight vector could be used in this class to perform a forward pass.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Tuple[int],
                 activation_fn: torch.nn.Module=torch.nn.ReLU(), 
                 has_activation_func_at_end: bool = False,
                 should_use_each_network_for_each_example: bool = False) -> None:
        """Initialises the main network.

        Args:
            in_dim (int): The dimension of the input data
            out_dim (int): The dimension of the output data, i.e. the predictions
            hidden_layers (Tuple[int]): A list/tuple of integers, representing the size of the hidden layers. 
                For instance, hidden_layers = [32, 32] would have weight matrices of shape (ignoring bias for illustration) 
                    (in_dim, 32), (32, 32), (32, out_dim)
            activation_fn (torch.nn.Module, optional): The activation function to use in between layers. Defaults to torch.nn.ReLU().
            has_activation_func_at_end (bool, optional): If True, the activation is also used at the end; by default it is not. Defaults to False.
            should_use_each_network_for_each_example (bool, optional): If this is True, then if we are given 10 networks and 20 examples, each network is evaluated on each example, to get 200 outputs as a result. If False, however, then we must have that in the forward function, we get either one network, or the same number of networks as examples, each network will be evaluated on its corresponding example. Defaults to False
                Hint: Use False for a Hypernetwork, and True for a Bayesian Neural Network.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layers = list(hidden_layers)
        self.activation_fn = activation_fn
        self.has_activation_func_at_end = has_activation_func_at_end
        self.should_use_each_network_for_each_example = should_use_each_network_for_each_example
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor):
        """This takes in data and weights, and performs a forward pass.
                We have two cases: 
                    - should_use_each_network_for_each_example == True
                        Here, x.shape == (x_batch_size, self.in_dim); weights.shape = (num_weight_samples, num_weights)
                        And we return something of shape (num_weight_samples, x_batch_size, self.out_dim)
                    
                    - should_use_each_network_for_each_example == False:
                        Here, x.shape == (x_batch_size, self.in_dim); weights.shape = (x_batch_size, num_weights)
                        And we return something of shape (x_batch_size, self.out_dim)

        Args:
            x (torch.Tensor): 
            weights (torch.Tensor): 

        Returns:
            _type_: 
        """
        
        x_batch_size = x.shape[0]
        num_weight_samples = weights.shape[0]
        assert weights.shape == (num_weight_samples, self.calc_num_weights_required())
        if not self.should_use_each_network_for_each_example:
            # Here, we must have that the number of networks is the same as the number of examples
            assert x_batch_size == num_weight_samples, f"{x_batch_size} != {num_weight_samples}"
            
        ws, bs = self.reshape_weights_to_correct_size(weights)
        
        # Prepare the Xs
        if self.should_use_each_network_for_each_example:
            # Change the shape of x from (batch_size, in_dim) to (num_weight_samples, batch_size, in_dim)
            # Use x * 1.0, as I am not sure how exactly the gradients work.
            x_large = []
            for i in range(num_weight_samples):
                x_large.append(x * 1.0)
            x = torch.stack(x_large)
            assert x.shape == (num_weight_samples, x_batch_size, self.in_dim)
        else:
            x = x.unsqueeze(1)
        
        # Transform the data
        for i, (w, b) in enumerate(zip(ws, bs)):
            if len(w) != len(x): assert False
            # This transpose is what torch.linear does!
            x = torch.bmm(x, torch.transpose(w, 1, 2)) + b
            if i < len(ws) - 1 or self.has_activation_func_at_end: x = self.activation_fn(x)
        
        if not self.should_use_each_network_for_each_example: x = x.squeeze(1) # remove the dummy added dimension
            
        y = x
        # Ensure the shapes are correct
        if self.should_use_each_network_for_each_example: assert y.shape == (num_weight_samples, x_batch_size, self.out_dim)
        else: assert y.shape == (x_batch_size, self.out_dim)
        return y

    
    def calc_num_weights_required(self) -> int:
        """Returns the number of weights required, i.e. the length of the parameter vector

        Returns:
            int: The number of weights
        """
        all_layers = self._all_layers()    
        tot = 0
        for i in range(len(all_layers) - 1):
            A, B = all_layers[i], all_layers[i+1]
            tot += (A+1) * B
        return tot
    
    def _all_layers(self):
        return [self.in_dim] + self.hidden_layers + [self.out_dim]
        
    def reshape_weights_to_correct_size(self, weights: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """This takes in a tensor of weights, with shape (num_weight_samples, num_weights_for_one_net) and returns new_weights, new_biases,
             where 
             new_weights is a list (one element for each layer), with shapes (batch_size, A, B), i.e. weight matrices for each layer
             new_biases is a list (one element for each layer), with shapes (batch_size, 1, B), i.e. basis vectors for each layer

        Args:
            weights (torch.Tensor): 

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 
        """
        new_weights = []
        new_biases  = []
        batch_size = weights.shape[0]
        all_layers = self._all_layers()
        CURR = 0
        assert len(all_layers) >= 2
        for i in range(len(all_layers) - 1):
            A, B = all_layers[i], all_layers[i+1]
            w = A * B
            # Need to reshape like this to ensure supervised learning does not fail.
            new_weights.append(weights[:, CURR:CURR + w].reshape(batch_size, B, A))
            CURR += w
            w = 1 * B
            new_biases.append(weights[:, CURR:CURR + w].reshape(batch_size, 1, B))
            CURR += w
        return new_weights, new_biases
    
    def __repr__(self) -> str:
        T = self.hidden_layers[0] if len(self.hidden_layers) else self.out_dim
        li = [f"Linear(in={self.in_dim}, out={T})"]
        sact = str(self.activation_fn)
        li.append(sact)
        for h in self.hidden_layers[1:-1]:
            li.append(f"Linear(in={T}, out={h})")
            li.append(sact)
            T = h
        li.append(f"Linear(in={T}, out={self.out_dim})")
        if self.has_activation_func_at_end:
            li.append(sact)
        sep = "\n  "
        s = sep.join(li)
        return f'MainNet({sep}{s}\n)'

def _test():
    K = 10 # number of networks
    B = 32 # batch size of X
    F = 4 # in dim
    O = 1 # out_dim
    H = 25 # Hidden
    main_net = MainNetwork(F, O, [H], should_use_each_network_for_each_example=True)
    torch.manual_seed(0)
    X = torch.randn((B, F))
    W = torch.randn((K, main_net.calc_num_weights_required()))
    
    Y = main_net.forward(X, W)
    print(Y.shape)
    
if __name__ == '__main__':
    _test()