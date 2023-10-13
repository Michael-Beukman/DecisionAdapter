import copy
from typing import Any, Dict, List, Tuple

import numpy as np
from genrlise.envs.base_env import BaseEnv
from genrlise.utils.types import Action, Context, State


class MonitorEnv(BaseEnv):
    """An Environment that saves the contexts, states and rewards for each episode. This is basically a wrapper
    """
    
    def __init__(self, base_env: BaseEnv, check_unwrapped: bool = True, do_save_all_transitions: bool = False, do_save_lengths=True, do_save_infos=False) -> None:
        """This monitors an environment

        Args:
            base_env (BaseEnv): The env to use
            check_unwrapped (bool, optional): If true, checks the unwrapped environment as well. Defaults to True.
            do_save_all_transitions (bool, optional): If true, saves all transitions. Defaults to False.
        """
        super().__init__([-20])
        self._check_unwrapped = check_unwrapped
        self.all_wrapped_contexts: List[Context] = []
        self.all_wrapped_init_states: List[State] = []
        self.all_episode_rewards: List[float] = []
        self.all_episode_lengths: List[int] = []
        self.all_episode_infos: List[dict] = []
        
        self.all_contexts: List[Context] = []
        self.all_init_states: List[State] = []
        
        self.do_save_all_transitions = do_save_all_transitions
        self.all_transitions = [[]]
        
        self.base_env = base_env
        self.do_save_lengths = do_save_lengths
        self.do_save_infos = do_save_infos
        self._curr_reward = 0
        self._curr_len = 0
        self._curr_infos = []
        self._good_keys = {
            'base_env', 'all_episode_rewards', 'all_init_states', 'all_contexts', '_curr_reward', 
            'all_wrapped_contexts', 'all_wrapped_init_states', '_check_unwrapped',
            'reset', '_reset', 'step', '_step', 'do_save_lengths', 'all_episode_lengths', '_curr_len',
            '_curr_infos', 'all_episode_infos', 'do_save_infos',
            '__getattribute__',
            'do_save_all_transitions', 'all_transitions'
        }
            
    def __getattribute__(self, name: str):
        og_env = object.__getattribute__(self, 'base_env')
        good_keys = object.__getattribute__(self, '_good_keys')
        if name in good_keys: return object.__getattribute__(self, name)
        return og_env.__getattribute__(name)

    def _reset(self) -> State:
        new_state = self.base_env.reset()
        if self.do_save_all_transitions:
            self.all_transitions[-1].append((new_state))
        
        self.all_wrapped_init_states.append(np.copy(new_state))
        self.all_episode_rewards.append(self._curr_reward)
        if self.do_save_lengths:
            self.all_episode_lengths.append(self._curr_len)
        if self.do_save_infos:
            self.all_episode_infos.append(self._curr_infos[-1:])
        self.all_wrapped_contexts.append(np.copy(self.base_env.get_context().detach().cpu().numpy()))
        self._curr_reward = 0
        self._curr_len = 0
        self._curr_infos = []
        
        if self._check_unwrapped:
            env = self.base_env
            while hasattr(env, 'original_env'):
                env = env.original_env
            unwrapped_state = env.get_state()
            unwrapped_context = env.get_context()
            
            self.all_contexts.append(np.copy(unwrapped_context))
            self.all_init_states.append(np.copy(unwrapped_state))
        
        return new_state

    def _step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        ns, r, d, i = self.base_env.step(action)
        if self.do_save_all_transitions:
            self.all_transitions[-1].append((np.copy(ns), r, d, copy.deepcopy(i), np.copy(action), np.copy(self.base_env.get_context().detach().cpu().numpy())))
            if d: self.all_transitions.append([])
        self._curr_reward += r
        self._curr_len += 1
        self._curr_infos.append(i)
        return ns, r, d, i
