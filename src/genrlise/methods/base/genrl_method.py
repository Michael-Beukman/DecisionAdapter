from typing import Any, Dict

import numpy as np
import torch
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.utils.types import EpisodeRewards, Metrics
from genrlise.contexts.problem import Problem


class GenRLMethod:
    """This is the method superclass, from which all methods in the genrlise/methods directory will inherit from.
    
    This is catered towards the multi-task / general setting, where we have a set of training tasks and a set of evaluation tasks.
    """
    def __init__(self, problem: Problem, device: torch.device, seed: np.random.SeedSequence, 
                 exp_conf: GenRLExperimentConfig, specific_kwargs: Dict[str, Any] = {}):
        self.env = problem.get_wrapped_env()
        self.problem = problem
        self.device = device
        self.seed = seed
        self.specific_kwargs = specific_kwargs
        if 'sb3_init_kwargs' not in self.specific_kwargs:
            self.specific_kwargs['sb3_init_kwargs'] = {}
        if 'device' not in self.specific_kwargs['sb3_init_kwargs']:
            self.specific_kwargs['sb3_init_kwargs']['device'] = device
        print("GenRLMethod:: Using Device:: ", self.specific_kwargs['sb3_init_kwargs']['device'])
        self.exp_conf = exp_conf
    
    
    def train(self, num_steps: int) -> Metrics:
        """This trains the current method, on `num_steps` steps in total, on the contexts given by self.problem

        Args:
            num_steps (int): How long to train for

        Returns:
            Metrics: The training metrics for this run, e.g. train reward, loss, etc.
        """
        pass
    
    def set_problem(self, problem: Problem) -> None:
        self.env = problem.get_wrapped_env()
        self.problem = problem

    def evaluate(self, number_of_episodes_total: int, test_problem: Problem) -> EpisodeRewards:
        """This should evaluate the trained model on `number_of_episodes` episodes on the testing context.

        Args:
            number_of_episodes_total (int): How many episodes to run this model for
            test_problem (Context): The problem to use

        Returns:
            EpisodeRewards: List of step rewards per episode.
        """
        pass
    
    def load_method_from_file(self, filename: str):
        """This should in principle load a method from a file, and use that in subsequent settings

        Args:
            filename (str):
        """
        raise NotImplementedError()
    