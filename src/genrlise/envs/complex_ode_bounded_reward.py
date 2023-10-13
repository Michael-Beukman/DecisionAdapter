from typing import Any, Dict, Tuple
import gym
import numpy as np
from numpy.random import default_rng

from genrlise.envs.complex_ode import ComplexODE
from genrlise.utils.types import Action, State

class ComplexODEBoundedReward(ComplexODE):
    """Complex ODE with a better reward -- a bounded value depending on if abs(state) is less than some specified value -- smaller is better.
    """
    def _get_reward(self, state: State, action: Action, next_state: State, done: bool) -> float:
        s = abs(next_state)
        
        bounds = [0.05, 0.1, 0.2, 0.5]
        for i, b in enumerate(bounds):
            if s < b: return 1 / (i + 1)
        if s < 2: return 0.05
        # Reward is either: 1, 0.5, 0.33, 0.25, 0.05 or 0
        return 0
    

if __name__ == '__main__':
    e = ComplexODE([1,1], 1)
    e.step([1, 1])