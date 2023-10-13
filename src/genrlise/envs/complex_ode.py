from typing import Any, Dict, Tuple
import gym
import numpy as np
from numpy.random import default_rng

from genrlise.envs.base_env import BaseEnv
from genrlise.utils.types import Action, State

class ComplexODE(BaseEnv):
    """
        This is a simple environment that follows a differential equation, parametrised by the context.
        Concretely it is defined as: 
        xdot = c0 * a + c1 * a^2 + c2 * a^3 + ..., where ci is the context.
        
        The difference between this and the above one is that here x dot does not depend on x.
    """
    def __init__(self, context: np.ndarray, seed: int, 
                 time_limit: int = 200,
                 force_mag: float = 1,
                 delta_time: float=  0.01,
                 max_x: float = 20,
                 use_only_single_action_dim = False,
                 normalise_obs: bool=False) -> None:
        """Initialises this environment
        
        Args:
            context (np.ndarray): The context variables
            seed (int): The seed to seed this environment with
            time_limit (int, optional): How long the environment goes on for. Defaults to 200.
            force_mag (float):    Maximum force magnitude
            delta_time (float):   Delta Time (x = x + xdot * dt)
            max_x (float):        Bounds on X
        """
        super().__init__(context)
        self.use_only_single_action_dim = use_only_single_action_dim
        self._context = context
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=((1,) if use_only_single_action_dim else (2,)))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.seed(seed)
        self.curr_reward = 0
        self.reset()
        self.time_limit = time_limit
        
        self.force_mag = force_mag
        self.delta_time = delta_time
        self.max_x = max_x
        self.normalise_obs = normalise_obs
        if self.normalise_obs:
            print("norm obs = ", self.normalise_obs)

    
    def _step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """This steps the simulation one step forward.

        Args:
            action (float): Continuous force between [-1, 1] #either 0, 1 or 2. This corresponds to applying a force of -FORCE_MAG, 0 or FORCE_MAG, respectively.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: _description_
        """
        force = action * self.force_mag
        old_state = np.copy(self.x)
        self._update_ode(force)
        self.step_count += 1
        done = self.is_done(old_state, action, self.x)
        r = self.get_reward(old_state, action, self.x, done)
        self.curr_reward += r
        info = {}
        if done: info['TimeLimit.truncated'] = True
        return [self.x, r, done, info]

    def _set_context(self, context):
        self._context = context

    def _set_state(self, state):
        self.x = state
    
    def _reset(self) -> Any:
        self.curr_reward = 0
        self.step_count = 0
        self.x = np.array(self.np_random.random() * 2 - 1).reshape(1, )
        return self.x
        
    def seed(self, seed=None):
        self.np_random = default_rng(seed)

    def _get_xdot(self, a) -> float:
        if self.use_only_single_action_dim:
            a = complex(a[0])
        else:
            a = complex(a[0], a[1])
        vals = []
        # vals = [a, x, ax, a^2]
        # https://stackoverflow.com/a/11299917
        vals = []
        for i, c in enumerate(self._context):
            vals.append(a ** (i + 1))
        
        xdot: complex = sum(c * v for c, v in zip(self._context, vals))
        xdot = float(xdot.real)
        return xdot
    
    def _get_state(self) -> State:
        if self.normalise_obs:
            return self.x / self.max_x
        return self.x
    
    def get_xdot(self, a) -> float:
        return self._get_xdot(a)

    def _update_ode(self, a: float):
        """This updates the ODE according to the equation, and given the action

        Args:
            a (float): _description_
        """
        xdot = self._get_xdot(a)
        self.x += xdot * self.delta_time
        self.x = np.clip(self.x, -self.max_x, self.max_x)
    
    def is_solvable(self):
        return - self._context[0] * self.force_mag < -self._context[1] * 1 and self._context[0] * self.force_mag > self._context[1] * 1

    def _is_done(self, state: State, action: Action, next_state: State) -> bool:
        return self.step_count >= self.time_limit
    
    def _get_reward(self, state: State, action: Action, next_state: State, done: bool) -> float:
        return float(- next_state ** 2)
    

if __name__ == '__main__':
    e = ComplexODE([1,1], 1)
    e.step([1, 1])