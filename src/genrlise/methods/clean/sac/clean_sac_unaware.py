import numpy as np
import torch
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.common.utils import maybe_save_cleanrl_model
from genrlise.contexts.problem import Problem
from genrlise.methods.base.genrl_method import GenRLMethod
from genrlise.methods.clean.sac.base_clean_sac import eval_sac, train_sac
from genrlise.utils.types import EpisodeRewards, Metrics
from typing import Any, Dict


class CleanSACUnaware(GenRLMethod):
    """Context Unaware SAC model, using the CleanRL code.
    """
    def __init__(self, problem: Problem, device: torch.device, seed: np.random.SeedSequence, exp_conf: GenRLExperimentConfig, int_seed: int, specific_kwargs: Dict[str, Any] = {}):
        super().__init__(problem, device, seed, exp_conf, specific_kwargs)
        self.model = None
        self.int_seed = int_seed
    
    def train(self, num_steps: int) -> Metrics:
        self.model = train_sac(num_steps, self.exp_conf, self.problem.get_wrapped_env(), problem=self.problem, device=self.device, int_seed=self.int_seed, model_starting_point=self.model,
                               actor_kwargs=self.exp_conf("method/args", {}), critic_kwargs=self.exp_conf("method/args", {}), checkpoint_path=self.specific_kwargs['log_dir'])
        maybe_save_cleanrl_model(self.model, self.specific_kwargs)
        return {}
    
    def evaluate(self, number_of_episodes_total: int, test_problem: Problem) -> EpisodeRewards:
        assert self.model is not None, "run .train() first before evaluating."
        return eval_sac(self.model, test_problem.get_wrapped_env(), number_of_episodes_total, self.exp_conf)
    
    def load_method_from_file(self, filename: str):
        self.model = torch.load(filename, map_location=self.device)