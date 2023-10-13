from typing import Any, Dict

import numpy as np
import torch
from external.repro_flap.sac_proper import eval_sac_flap_canonical, train_sac_flap_proper
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.common.utils import maybe_save_cleanrl_model
from genrlise.contexts.problem import Problem
from genrlise.methods.base.genrl_method import GenRLMethod

from genrlise.utils.types import EpisodeRewards, Metrics


class CleanSACCanonicalFLAP(GenRLMethod):
    """
        This is Peng et al 2021, including the altered learning procedure.
    """

    def __init__(self, problem: Problem, device: torch.device, seed: np.random.SeedSequence, int_seed: int, exp_conf: GenRLExperimentConfig, specific_kwargs: Dict[str, Any] = ...):
        super().__init__(problem, device, seed, exp_conf, specific_kwargs)
        self.int_seed = int_seed
        self.model = None

    
    def _context_dimension(self) -> int:
        return self.env.encoder.context_dimension

    def _get_learn_function(self):
        return train_sac_flap_proper
    
    def _get_learn_kwargs(self):
        return {}
    
    def train(self, num_steps: int) -> Metrics:
        # Now I need to get a list of envs instead of one env.
        all_envs_to_use = self.problem.get_list_of_envs()
        
        KK = 105
        actor_kwargs = {
            'net_arch': self.exp_conf("method/args/actor_net_arch", [KK, KK, KK])
        }
        
        critic_kwargs = {
            'net_arch': self.exp_conf("method/args/critic_net_arch", [KK, KK, KK])
        }
                
        adapter_kwargs = {
            'net_arch': self.exp_conf("method/args/adapter_net_arch", [2*KK, 2*KK, 2*KK]),
        }

        F = self._get_learn_function()
        if num_steps == 0 and self.model is not None:
            print("Skipping the train call due to model being not none and timesteps = 0")
            pass
        else:
            self.model = F(num_steps, self.exp_conf, all_envs_to_use, context_dimension=self._context_dimension(), device=self.device, int_seed=self.int_seed, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs, model_starting_point=self.model, adapter_kwargs=adapter_kwargs, checkpoint_path=self.specific_kwargs['log_dir'], **self._get_learn_kwargs())
    
        maybe_save_cleanrl_model(self.model, self.specific_kwargs)
        return {}
    
    def evaluate(self, number_of_episodes_total: int, test_problem: Problem) -> EpisodeRewards:
        adapt_steps = self.exp_conf("method/args/adaptation_steps", 30)
        use_ctx_adapt   = self.exp_conf("method/args/use_context_adapter_to_eval", False)
        wrap = test_problem.get_wrapped_env()
        assert self.model is not None, "run .train() first before evaluating."
        return eval_sac_flap_canonical(self.model, wrap, number_of_episodes_total, self.exp_conf, use_adapter=True, adaptation_steps=adapt_steps, use_context_adapter=use_ctx_adapt)
    
    
    def __repr__(self) -> str:
        s = f"FlapCanonical"
        return s
    
    def load_method_from_file(self, filename: str):
        self.model = torch.load(filename, map_location=self.device)
        self.model['device'] = self.device
