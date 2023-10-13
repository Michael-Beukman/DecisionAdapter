from typing import Any, Dict, Tuple
import numpy as np
from genrlise.contexts.context_sampler import ContextSampler
from genrlise.envs.base_env import BaseEnv
from genrlise.utils.types import Action, State


class ContextSamplerWrapperEnv(BaseEnv):
    """This is a wrapper around a BaseEnv, where we sample a different context on each episode. This is useful for an API that just takes in a gym environment, e.g. stable baselines
    """
    def __init__(self, base_env: BaseEnv, context_sampler: ContextSampler) -> None:
        super().__init__([-10])
        self.original_env = base_env
        self.sampler = context_sampler
        self.init_states = []
        
        self._good_keys = {
            'original_env', 'sampler', 'init_states', 'reset', 'step',
            '__getattribute__', 'reset', '_reset', 'step', '_step',
            
            '_episode_length'
        }
        self._episode_length: int = 0
        
    def _reset(self) -> State:
        # Resets and sets the context of the base environment
        self.sampler.set_recent_episode_length(self._episode_length)
        self.original_env.set_context(self.sampler.sample_context())
        self._episode_length  = 0
        init_state = self.original_env.reset()
        if self.sampler.can_sample_state:
            init_state = self.sampler.sample_state()
            self.original_env._set_state(init_state)
        self.init_states.append(np.copy(init_state))
        return init_state
    
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
    
    def _step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        self._episode_length += 1
        return self.original_env.step(action)
    
