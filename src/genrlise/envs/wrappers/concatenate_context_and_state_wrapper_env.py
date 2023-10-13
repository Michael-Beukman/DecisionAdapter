from typing import List, Union
import gym
import numpy as np
from genrlise.envs.base_env import BaseEnv

def _get_proper_context_norm_vals(given, default, context_dimension):
    if given is None:
        given = [default] * context_dimension
    elif type(given) == float or type(given) == int:
        given = [given] * context_dimension
    elif type(given) == list:
        pass
    else:
        print(f"Invalid value ({given}) -- {type(given)}")
    
    assert len(given) == context_dimension
    return given

class ConcatenateContextAndStateWrapperEnv(BaseEnv):
    """This concatenates the context and state together, and gives that as the state to the agent.

    Args:
        BaseEnv (_type_): _description_
    """
    def __init__(self, base_env: BaseEnv, context_dimension: int, context_normalisations_low: Union[int, List[int]] = None, context_normalisations_high: Union[float, List[float]] = None) -> None:
        super().__init__([-10])
        self.original_env = base_env
        self._good_keys = {
            'original_env', 'encoder', 'get_context', 'reset', 'step', 'concat', '_good_keys',
            '__getattribute__', 'observation_space', 'get_context', '_reset', '_step'
        }
        shape = self.original_env.observation_space.shape
        assert len(shape) == 1 or len(shape) == 2 and shape[-1] == 1, f"Bad shape {shape}"
        shape = (shape[0] + context_dimension,)
        low, high = self.base_env.observation_space.low, self.base_env.observation_space.high
        
        context_normalisations_low  = _get_proper_context_norm_vals(context_normalisations_low, -np.inf, context_dimension)
        context_normalisations_high = _get_proper_context_norm_vals(context_normalisations_high, np.inf, context_dimension)

        low = np.append(low.flatten(), context_normalisations_low)
        high = np.append(high.flatten(), context_normalisations_high)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
        

        
    def _reset(self):
        state = self.original_env.reset()
        return self.concat(state)

    def _step(self, action):
        state, r, d, i = self.original_env.step(action)
        return self.concat(state), r, d, i
    
        
    def __getattribute__(self, attr):
        # Pass through all other calls to the base env.
        dict = object.__getattribute__(self, '__dict__')
        goods = object.__getattribute__(self, '_good_keys')
        
        if attr in goods:
            return object.__getattribute__(self, attr)
    
        if 'original_env' not in dict:
            raise AttributeError
        og = object.__getattribute__(self, 'original_env')
        return getattr(og, attr)


    def concat(self, state):
        if len(state.shape) == 2 and state.shape[-1] == 1:
            s = state[:, 0]
        else:
            s = state
        try:
            ctx = self.original_env.get_context().cpu().numpy()
            return np.concatenate([s, ctx])
        except Exception as e:
            raise e
