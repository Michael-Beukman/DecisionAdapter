import copy
from typing import Any, Dict, Tuple, final
import gym
from genrlise.utils.types import Action, Context, State

class BaseEnv(gym.Env):
    # A base environment class to deal with different contexts and settings. This is basically the standard gym interface, but with the ability to set a context, and some other general utilities.
    def __init__(self, init_context: Context) -> None:
        super().__init__()
        self.episode_count = 0 
        self._context = init_context
    
    @final
    def reset(self) -> State:
        self.episode_count += 1
        return self._reset()

    
    @final
    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        return self._step(action)
    
    
    def get_context(self) -> Context:
        return self._context
    
    @final
    def set_context(self, context: Context):
        return self._set_context(copy.deepcopy(context))

    
    @final
    def get_state(self) -> State:
        return self._get_state()

    @final
    def set_state(self, state: State):
        return self._set_state(state)
        
    @final
    def get_reward(self, state: State, action: Action, next_state: State, done: bool) -> float:
        return self._get_reward(state, action, next_state, done)
    
    @final
    def is_done(self, state: State, action: Action, next_state: State) -> bool:
        return self._is_done(state, action, next_state)

    
    
    
    # To be implemented by subclasses
    
    
    def _reset(self) -> State:
        raise NotImplementedError()
    
    
    def _step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        raise NotImplementedError()


    def _get_state(self) -> State:
        raise NotImplementedError()
    
    def _set_context(self, context: Context):
        raise NotImplementedError()

    def _get_reward(self, state: State, action: Action, next_state: State, done: bool) -> float:
        raise NotImplementedError()

    def _is_done(self, state: State, action: Action, next_state: State) -> bool:
        raise NotImplementedError()
    
    def _set_state(self, state: State):
        raise NotImplementedError()

