import numpy as np
from stable_baselines3 import SAC
import torch
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.contexts.problem import Problem
from genrlise.methods.base.genrl_method import GenRLMethod
from genrlise.utils.types import EpisodeRewards, Metrics
from typing import Any, Dict



class OptimalODEModel(GenRLMethod):
    """ The optimal ODE model, to be able to compare other approaches against this.
    """
    def __init__(self, problem: Problem, device: torch.device, seed: np.random.SeedSequence, exp_conf: GenRLExperimentConfig, specific_kwargs: Dict[str, Any] = {}, MAX_FORCE_MAG: float = 1):
        super().__init__(problem, device, seed, exp_conf, specific_kwargs)
        self.model: SAC = None
        self.MAX_FORCE_MAG = MAX_FORCE_MAG
        print("Optimal ODE Method max force mag = ", self.MAX_FORCE_MAG)
    
    def train(self, num_steps: int) -> Metrics:
        env = self.problem.get_wrapped_env()
        current_t = 0
        all_rs = []
        contexts = []
        while current_t < num_steps:
            rs = []
            x = env.reset()
            init_state = np.array(np.copy(x))
            init_state = init_state.flatten()
            done = False
            contexts.append(env.get_context().detach().cpu().numpy())
            while not done:
                current_t += 1
                action = self._get_optimal_action(env)
                x, r, done, _ = env.step(action)
                rs.append(r)
            C = contexts[-1].flatten()
            all_rs.append(rs)
        env.reset()
        
        return {
            'all_rewards': all_rs
        }


    def evaluate(self, number_of_episodes_total: int, test_problem: Problem) -> EpisodeRewards:
        self.set_problem(test_problem)
        a = self.train(number_of_episodes_total * self.env.time_limit)
        a = np.sum(a['all_rewards'], axis=-1)
        return a
    
    def _get_optimal_action(self, env = None, x = None) -> np.ndarray:
        if env is None: env = self.env
        return self._get_action_given_context(env.get_context().detach().cpu().numpy(), env, x)
        
    def _get_action_given_context(self, context, env = None, x = None) -> np.ndarray:
        if env is None: env = self.env
        if x is None: x = env.x
        
        _TIME = env.delta_time
        _FORCE_MAG = env.force_mag
        poly = np.polynomial.Polynomial(
            [np.array(x).squeeze() / _TIME] + [a for i, a in enumerate(list(context))]
        )
        roots = (poly.roots()) / _FORCE_MAG
        KK = self.MAX_FORCE_MAG
        best_dist = np.inf
        best_root = 0
        def get_ans(r):
            if np.iscomplex(r): 
                return np.clip([r.real, r.imag], -KK, KK)
            else: 
                return np.clip([r, 0], -KK, KK)
        
        for i in range(len(roots)):
            r = roots[i]
            A = get_ans(r)
            new_x = x + env.get_xdot(A) * _TIME
            if abs(new_x) < best_dist:
                best_dist = abs(new_x)
                best_root = i
        if len(roots) == 0: return [0, 0]
        ans = get_ans(roots[best_root])
        return ans
