from typing import Any, Dict, List, Union
import gym
import numpy as np
import torch
from numpy.random import default_rng

from genrlise.envs.base_env import BaseEnv

class ContextEncoder:
    """
        This is a context encoder: A class that sits between the environment and the agent. It has one main method to use, `get_context()` which returns the context.
    """
    def __init__(self, env: BaseEnv, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed = None, specific_dimensions_to_use: Union[List[int], None] = None, **kwargs) -> None:
        # `specific_dimensions_to_use`: If not None, only use these dimensions in the context.
        self.specific_dimensions_to_use = specific_dimensions_to_use
        self.env = env
        self.context_dimension = context_dimension
        self.device = device
        self.seed = seed
        if self.seed is not None:
            self.np_random = default_rng(seed)
        else:
            self.np_random = None
        
        self._prev_ctx = None
    
    def add_transition(self, state, action, next_state):
        pass
    
    def _my_get_context(self):
        ans = self._get_context()
        if self.specific_dimensions_to_use is not None:
            ans = ans[self.specific_dimensions_to_use]
        
        return ans
    
    def get_context(self, return_prev_ctx: bool = False) -> torch.Tensor:
        if return_prev_ctx:
            ans = self._prev_ctx
            self._prev_ctx = self._my_get_context()
            return ans
        if self._prev_ctx is None: self._prev_ctx = self._my_get_context()
        return self._my_get_context()
        
    
    def _get_context(self) -> torch.Tensor:
        raise NotImplementedError("Implement This")

    def to(self, device: torch.device) -> "ContextEncoder":
        self.device = device
        return self

    def set_env(self, env: BaseEnv):
        self.env = env
        
    def set_eval(self):
        pass

class DefaultContextEncoder(ContextEncoder):
    """Default context encoder: Returns a list of ones

    Args:
        ContextEncoder (_type_): _description_
    """
    def __init__(self, env: BaseEnv, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed=None, default_value=1, **kwargs) -> None:
        super().__init__(env, context_dimension, device, seed=seed, **kwargs)
        self.default_value = default_value
        
    def _get_context(self) -> torch.Tensor:
        return torch.ones(self.context_dimension).to(self.device) * self.default_value
    
class ConstantContextEncoder(ContextEncoder):
    """Returns a constant list of contexts

    Args:
        ContextEncoder (_type_): _description_
    """
    def __init__(self, env: BaseEnv, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed=None, values=[1, 1], **kwargs) -> None:
        super().__init__(env, context_dimension, device, seed=seed, **kwargs)
        self.values = values
        
    def _get_context(self) -> torch.Tensor:
        return torch.tensor(self.values).to(self.device)

class RandomNormalContextEncoder(ContextEncoder):
    def __init__(self, env: BaseEnv, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed=None, **kwargs) -> None:
        super().__init__(env, context_dimension, device, seed=seed, **kwargs)
        self.A = None
        
    def _get_context(self) -> torch.Tensor:
        if self.A is None:
            self.A = torch.tensor(self.np_random.standard_normal(self.context_dimension)).to(self.device).float()
        return self.A
    


class GroundTruthContextEncoder(ContextEncoder):
    def __init__(self, env: BaseEnv, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed=None, **kwargs) -> None:
        super().__init__(env, context_dimension, device, seed=seed)
    
    def _get_context(self) -> torch.Tensor:
        ctx = self.env.get_context()
        return torch.Tensor(ctx).to(self.device)


class GroundTruthNormalisedContextEncoder(ContextEncoder):
    def __init__(self, env: BaseEnv, context_dimension: int = 2, device: torch.device = torch.device('cpu'), min=-1, max=1, normalise_mode='zero_one', **kwargs) -> None:
        super().__init__(env, context_dimension, device, **kwargs)
        self.min = torch.Tensor(min).to(device)
        self.max = torch.Tensor(max).to(device)
        self.normalise_mode = normalise_mode
        print(f"Creating normalised ground truth context encoder with values min={min}, max={max}")
    
    def _get_context(self) -> torch.Tensor:
        if self.min.device != self.device:
            self.min = self.min.to(self.device)
            self.max = self.max.to(self.device)
        ctx = self.env.get_context()
        ans = torch.Tensor(ctx).to(self.device)
        if self.normalise_mode == 'zero_one':
            new = (ans - self.min) / (self.max - self.min)
        elif self.normalise_mode == 'negone_zero' or self.normalise_mode == 'negone_one':
            new = 2 * (ans - self.min) / (self.max - self.min) - 1
        else: assert False, f"Mode {self.normalise_mode} is not supported"  
        return new

class NoisyContextEncoder(ContextEncoder):
    def __init__(self, env: BaseEnv, encoder: ContextEncoder, noise: Dict[str, Any] = {'type': 'gaussian', 'sigma': 0.2}, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed=None, specific_dimensions_to_use: Union[List[int], None] = None) -> None:
        super().__init__(env, context_dimension, device, seed, specific_dimensions_to_use)
        self.encoder = encoder
        self.noise_profile = noise
    
    def _get_context(self) -> torch.Tensor:
        ans = self.encoder.get_context()
        assert self.noise_profile['type'] == 'gaussian'
        return ans + torch.randn_like(ans) * self.noise_profile['sigma']


class NoisyConsistentContextEncoder(ContextEncoder):
    # Consistently have a single context, which is noisy but constant, per episode instead of adding noise for each step
    def __init__(self, env: BaseEnv, encoder: ContextEncoder, noise: Dict[str, Any] = {'type': 'gaussian', 'sigma': 0.2}, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed=None, specific_dimensions_to_use: Union[List[int], None] = None) -> None:
        super().__init__(env, context_dimension, device, seed, specific_dimensions_to_use)
        self.encoder = encoder
        self.noise_profile = noise
        self.prev_count = self.encoder.env.episode_count
        self.random_val = None
        assert self.noise_profile['type'] == 'gaussian'
    
    def _get_context(self) -> torch.Tensor:
        test_ctx = tuple(self.encoder.get_context().cpu().detach().numpy().tolist())
        
        og = self.encoder.get_context()
        if self.encoder.env.episode_count != self.prev_count: # new episode, so change the random value
            self.prev_count = self.encoder.env.episode_count
            self.random_val = torch.randn_like(og) * self.noise_profile['sigma']
        
        ans = og + self.random_val
        
        return ans
        

class DummyDimensionsContextEncoder(ContextEncoder):
    # Have N additional dimensions that change somewhat regularly.
    def __init__(self, env: BaseEnv, encoder: ContextEncoder, context_dimension: int = 2, device: torch.device = torch.device('cpu'), seed=None, specific_dimensions_to_use: Union[List[int], None] = None,
                 extra_dimensions: int = 3, change_val = 0.5, fixed_vals: list[float] = None) -> None:
        super().__init__(env, context_dimension, device, seed, specific_dimensions_to_use)
        self.encoder = encoder
        self.change_val = change_val
        self.extra_dimensions = extra_dimensions
        self.prev_count = self.encoder.env.episode_count
        self.random_val = None
        if type(fixed_vals) == int or type(fixed_vals) == float:
            fixed_vals = [fixed_vals for _ in range(extra_dimensions)]
        self.fixed_vals = fixed_vals
        
        if fixed_vals is None:
            self.extras = torch.ones(extra_dimensions).to(device)
        else:
            self.extras = torch.tensor(fixed_vals).to(device)
            
        self._curr_extra_index = 0
    
    def _get_context(self) -> torch.Tensor:
        og = self.encoder.get_context()
        if self.encoder.env.episode_count != self.prev_count: # new episode, so change the random value
            self.prev_count = self.encoder.env.episode_count
            if self.fixed_vals is None:
                self.extras = (self.extras).clone()
                self.extras[self._curr_extra_index] -= self.change_val
                if self.extras[self._curr_extra_index] < 0:
                    self.extras[self._curr_extra_index] = 1.0
                    self._curr_extra_index += 1
                    self._curr_extra_index = (self._curr_extra_index % len(self.extras))
        
        ans = torch.cat([og, self.extras])
        return ans
        


class DummyDimensionGaussianContextEncoderWithMean(ContextEncoder):
    # Consistently have a single context, which is noisy but constant, per episode instead of adding noise for each step
    def __init__(self, env: BaseEnv, encoder: ContextEncoder, noise: Dict[str, Any] = {'type': 'gaussian', 'sigma': 0.2, 'mean': 0}, context_dimension: int = 2, extra_dimensions: int = 1, device: torch.device = torch.device('cpu'), seed=None, specific_dimensions_to_use: Union[List[int], None] = None) -> None:
        super().__init__(env, context_dimension, device, seed, specific_dimensions_to_use)
        self.encoder = encoder
        self.noise_profile = noise
        self.prev_count = self.encoder.env.episode_count
        self.random_val = None
        self.extra_dimensions = extra_dimensions
        assert self.noise_profile['type'] == 'gaussian'
    
    def _get_context(self) -> torch.Tensor:
        test_ctx = tuple(self.encoder.get_context().cpu().detach().numpy().tolist())
        
        og = self.encoder.get_context()
        if self.encoder.env.episode_count != self.prev_count: # new episode, so change the random value
            self.prev_count = self.encoder.env.episode_count
            self.random_val = (torch.randn(self.extra_dimensions) * self.noise_profile['sigma'] + self.noise_profile['mean']).to(self.device)
        
        ans = torch.cat([og, self.random_val])
        return ans
        
