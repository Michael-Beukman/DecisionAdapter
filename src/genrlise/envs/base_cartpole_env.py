"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.utils import seeding
from numpy.random import default_rng

from genrlise.envs.base_env import BaseEnv
from genrlise.utils.types import Action, State

_DEFAULT_CONTEXT = [
    9.8,       #  gravity
    1.0,       #  masscart
    0.1,       #  masspole
    0.5,       #  length
    10.0,      #  force_mag
]

_STEP_MAX = 500

class BaseCartPoleEnv(BaseEnv):
    """
    2022/04/16: Copy the code from here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    To make an environment that can handle (1) continuous actions and (2) a parametrisable context
    
    
    ### Description
    This environment corresponds to the version of the cart-pole problem
    described by Barto, Sutton, and Anderson in ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a
    frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Cart Position         | -4.8                 | 4.8                |
    | 1   | Cart Velocity         | -Inf                 | Inf                |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
    | 3   | Pole Angular Velocity | -Inf                 | Inf                |
    **Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken, including the termination step, is allotted. The threshold for rewards is 475 for v1.
    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    ### Episode Termination
    The episode terminates if any one of the following occurs:
    1. Pole Angle is greater than ±12°
    2. Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Episode length is greater than 500 (200 for v0)
    ### Arguments
    ```
    gym.make('CartPole-v1')
    ```
    No additional arguments are currently supported.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50,
                "render.modes": ["human", "rgb_array"], "render.fps": 50}
    DEFAULT_CONTEXT = _DEFAULT_CONTEXT
    def __init__(self,
                 context=_DEFAULT_CONTEXT, 
                 seed = None,
                 flatten_state: bool = False,
                 continuous_actions: bool = True, 
                 infinite_actions: bool = False,
                 large_actions: bool = False,
                 action_magnitude: int = None,
                 
                 have_state_be_x_y_position: bool = False,
                 
                 max_timesteps: int = _STEP_MAX
                 ):
        super().__init__(context)
        
        good_num = 4
        if have_state_be_x_y_position: good_num = 5
        self.GOOD_OBS_SHAPE = (good_num, 1) if not flatten_state else (good_num,)
        self.continuous_actions = continuous_actions
        self._context = context
        self.gravity = context[0]
        self.masscart = context[1]
        self.masspole = context[2]
        self.length = context[3]
        self.force_mag = context[4]
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self.have_state_be_x_y_position = have_state_be_x_y_position
        self.curr_reward = 0
        self.max_timesteps = max_timesteps
        print("MAX timesteps = ", max_timesteps)
        if seed is not None:
            self.seed(seed)



        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        
        
        if have_state_be_x_y_position:
            
            high = np.array(
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    
                    np.sin(self.theta_threshold_radians * 2) * 10,
                    np.cos(self.theta_threshold_radians * 2) * 10,
                    
                    np.finfo(np.float32).max,
                ],
                dtype=np.float32,
            ).reshape(*self.GOOD_OBS_SHAPE)
        else:
            high = np.array(
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians * 2,
                    np.finfo(np.float32).max,
                ],
                dtype=np.float32,
            ).reshape(*self.GOOD_OBS_SHAPE)

        self.infinite_actions = infinite_actions
        self.large_actions = large_actions
        self.action_magnitude = action_magnitude
        if self.continuous_actions:
            if action_magnitude is not None:
                self.action_space = spaces.Box(low=-action_magnitude, high=action_magnitude, shape=(1,), dtype=np.float32)
            else:
                if large_actions:
                    self.action_space = spaces.Box(low=-10000, high=10000, shape=(1,), dtype=np.float32)
                elif infinite_actions:
                    self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
                else:
                    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32, shape=self.GOOD_OBS_SHAPE)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None
        self.stepcount = 0
        self.flatten_state = flatten_state

    def _set_context(self, context):
        self._context = context
        self.gravity = context[0]
        self.masscart = context[1]
        self.masspole = context[2]
        self.length = context[3]
        self.force_mag = context[4]
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def _set_state(self, state):
        self.state = state

    def _step(self, action):
        self.stepcount += 1
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        if self.have_state_be_x_y_position:
            (x, x_dot, x_pos, y_pos, theta_dot) = self.state
            theta = np.arcsin(x_pos / self.length)
        else:
            x, x_dot, theta, theta_dot = self.state
        if self.continuous_actions:
            force = self.force_mag * action
        else:
            force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        old_state = self.state
        
        if self.have_state_be_x_y_position:
            x_pos = np.sin(theta) * self.length
            y_pos = np.cos(theta) * self.length
            self.state = (x, x_dot, x_pos, y_pos, theta_dot)
        else:
            self.state = (x, x_dot, theta, theta_dot)
        done = self.is_done(old_state, action, self.state)

        reward = self.get_reward(old_state, action, self.state, done)
        self.curr_reward += reward
        _new_s = np.array(self.state, dtype=np.float32)
        if self.flatten_state: _new_s = _new_s.reshape(-1)
        info = {}
        if done and self.stepcount >= self.max_timesteps: info['TimeLimit.truncated'] = True
        return _new_s, reward, done, info

    def _reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        self.curr_reward = 0
        self.stepcount = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,1) if not self.flatten_state else (4,))
        
        if self.have_state_be_x_y_position:
            theta = self.state[2]
            x_pos = np.sin(theta) * self.length
            y_pos = np.cos(theta) * self.length
            self.state = np.array([self.state[0], self.state[1], x_pos, y_pos, self.state[3]])
        
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
    
    def seed(self, seed=None):
        self.np_random = default_rng(seed)
        return 
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        import pygame
        from pygame import gfxdraw

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def _get_state(self) -> State:
        return self.state
    
    def _is_done(self, state: State, action: Action, next_state: State) -> bool:
        if self.have_state_be_x_y_position:
            (x, x_dot, x_pos, y_pos, theta_dot) = next_state
            theta = np.arcsin(x_pos / self.length)
        else:
            (x, x_dot, theta, theta_dot) = next_state
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians

            or self.stepcount >= self.max_timesteps
        )
    
    def _get_reward(self, state: State, action: Action, next_state: State, done: bool) -> float:
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        return reward
    
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False