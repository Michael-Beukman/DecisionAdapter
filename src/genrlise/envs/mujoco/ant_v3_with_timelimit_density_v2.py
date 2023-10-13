import os
import numpy as np

from gym import utils
from gym.spaces import Box
from genrlise.envs.base_env import BaseEnv
from gym import spaces
from typing import Any, Dict, Optional, Tuple, Union

from genrlise.envs.mujoco.base_mujoco_env import MujocoEnv
from genrlise.utils.types import Action, Context, State

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class AntEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file,
        seed,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        time_limit=1000
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.time_limit = time_limit
        self.curr_step = 0
        MujocoEnv.__init__(self, xml_file, 5, seed=seed)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        if self.curr_step >= self.time_limit: done = True
        return done

    def step(self, action):
        self.curr_step += 1
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        self.curr_step = 0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

def _setup_envs(seed, int_seed):
    # Create environments, as each one with has a different physics engine.
    allowed_contexts = [[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5], [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5], [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5], [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0], [20.5], [21.0], [21.5], [22.0], [22.5], [23.0], [23.5], [24.0], [24.5], [25.0], [25.5], [26.0], [26.5], [27.0], [27.5], [28.0], [28.5], [29.0], [29.5], [30.0], [30.5], [31.0], [31.5], [32.0], [32.5], [33.0], [33.5], [34.0], [34.5], [35.0], [35.5], [36.0], [36.5], [37.0], [37.5], [38.0], [38.5], [39.0], [39.5], [40.0], [40.5], [41.0], [41.5], [42.0], [42.5], [43.0], [43.5], [44.0], [44.5], [45.0], [45.5], [46.0], [46.5], [47.0], [47.5], [48.0], [48.5], [49.0], [49.5], [50.0], [50.5], [51.0], [51.5], [52.0], [52.5], [53.0], [53.5], [54.0], [54.5], [55.0], [55.5], [56.0], [56.5], [57.0], [57.5], [58.0], [58.5], [59.0], [59.5], [60.0], [60.5], [61.0], [61.5], [62.0], [62.5], [63.0], [63.5], [64.0], [64.5], [65.0], [65.5], [66.0], [66.5], [67.0], [67.5], [68.0], [68.5], [69.0], [69.5], [70.0], [70.5], [71.0], [71.5], [72.0], [72.5], [73.0], [73.5], [74.0], [74.5], [75.0], [75.5], [76.0], [76.5], [77.0], [77.5], [78.0], [78.5], [79.0], [79.5], [80.0], [80.5], [81.0], [81.5], [82.0], [82.5], [83.0], [83.5], [84.0], [84.5], [85.0], [85.5], [86.0], [86.5], [87.0], [87.5], [88.0], [88.5], [89.0], [89.5], [90.0], [90.5], [91.0], [91.5], [92.0], [92.5], [93.0], [93.5], [94.0], [94.5], [95.0], [95.5], [96.0], [96.5], [97.0], [97.5], [98.0], [98.5], [99.0], [99.5], [100.0]]
    filenames = ["src/genrlise/envs/mujoco/xmls/ant_density_v2/ant_{}.xml".format(str(num[0]).replace(".", "_")) for num in allowed_contexts]
    a = {}
    for filename, context in zip(filenames, allowed_contexts):
        absp = os.path.abspath(filename)
        env = AntEnv(absp, seed=int_seed)
        environment_kwargs = {}
        c = tuple([round(context[0] * 10000)] )

        a[c] = env
    return a




class MujocoAntEnvTimeLimitDensityV2(BaseEnv):
    """Ant
    """
    DEFAULT_CONTEXT = [5.0]
    def __init__(self, context: Context, seed, int_seed, flatten_state: bool = False) -> None:
        super().__init__(context)
        self.seed = seed
        self.flatten_state = flatten_state
        self._all_envs = _setup_envs(seed, int_seed)
        assert self.get_context_hash() in self._all_envs
        self.curr_env = self._all_envs[self.get_context_hash()]
        for c, e in self._all_envs.items():
            e.action_space.seed(int_seed)
        
        self.action_space = self.curr_env.action_space
        self.observation_space = self.curr_env.observation_space
        self._state = 0
    
    def _process_state(self, state):
        if self.flatten_state: state = state.reshape(-1)
        return state

    def _step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        ans = self.curr_env.step(action)
        
        ans = (self._process_state(ans[0]), ans[1], ans[2], ans[3])
        self._state = ans[0]
        return ans
    
    def _reset(self) -> State:
        s = self.curr_env.reset()
        s = self._process_state(s)
        self._state = s
        return s

    def _get_state(self) -> State:
        return self._state
    
    def _set_context(self, context: Context):
        self._context = np.array(context, dtype=np.float32)
        self.curr_env = self._all_envs[self.get_context_hash()]
    
    def _get_reward(self, state: State, action: Action, next_state: State, done: bool) -> float:
        raise NotImplementedError
    
    def _is_done(self, state: State, action: Action, next_state: State) -> bool:
        raise NotImplementedError
    
    def _set_state(self, state: State):
        raise NotImplementedError
    
    def get_context_hash(self):
        AA = tuple(map(int, np.round(np.array(self.get_context()) * 10000).flatten().tolist()))
        return AA
    
    def render(self):
        self.curr_env.render()
