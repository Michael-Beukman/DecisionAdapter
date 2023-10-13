import copy
from typing import Any, Dict

import numpy as np
import torch
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.common.utils import maybe_save_cleanrl_model
from genrlise.contexts.problem import Problem
from genrlise.envs.wrappers.concatenate_context_and_state_wrapper_env import ConcatenateContextAndStateWrapperEnv
from genrlise.methods.base.genrl_method import GenRLMethod
from genrlise.methods.clean.sac.base_clean_sac import eval_sac, train_sac
from genrlise.methods.clean.sac.networks.adapter import ActorAdapterOnlyMean, SoftQNetworkAdapter
from genrlise.utils.types import EpisodeRewards, Metrics


class CleanSACAdapterV1(GenRLMethod):
    """
        This uses a standard main network, and a hypernetwork to generate an adapter.
    """

    def __init__(
        self,
        problem: Problem,
        device: torch.device,
        seed: np.random.SeedSequence,
        int_seed: int,
        exp_conf: GenRLExperimentConfig,
        specific_kwargs: Dict[str, Any] = ...,
        
        pretrain_steps: int = None,
        freeze_adapter_during_pretraining: bool=False,
        freeze_main_net_initial_during_fine_tune: bool = False,
        freeze_main_net_final_during_fine_tune: bool = False,
        
        should_enable_disable_adapter: bool = False,
        use_segmented_adapter: bool = False,
        use_segmented_modulator: bool = False,
    ):
        """This creates the adapter model. The main idea behind this is that we have a specific network, and an adapter that is added in between the layers of this main network.

        Args:
            problem (Problem): The problem to train on
            device (torch.device): cpu or gpu
            seed (np.random.SeedSequence): The seed to use for random initialisations
            exp_conf (GenRLExperimentConfig): An experiment configuration to use
            specific_kwargs (Dict[str, Any], optional): . Defaults to ....
            pretrain_steps (int, optional): The number of steps to pretrain. If None, then we do not pretrain. Defaults to None.
            
            freeze_adapter_during_pretraining (bool, optional): Only available if we do pre-training. If this is true, we freeze the adapter during pre-training. Defaults to False.
            freeze_main_net_initial_during_fine_tune (bool, optional): If true, we freeze the main network before the adapter during fine-tuning. Defaults to False.
            freeze_main_net_final_during_fine_tune (bool, optional): If true, we freeze the main network after the adapter during fine-tuning. Defaults to False.
        """
        super().__init__(problem, device, seed, exp_conf, specific_kwargs)
        self.do_pretraining = pretrain_steps is not None
        self.pretrain_steps = pretrain_steps
        self.freeze_adapter_during_pretraining = freeze_adapter_during_pretraining
        
        self.freeze_main_net_initial_during_fine_tune = freeze_main_net_initial_during_fine_tune
        self.freeze_main_net_final_during_fine_tune = freeze_main_net_final_during_fine_tune
        self.should_enable_disable_adapter = should_enable_disable_adapter
        self.use_segmented_adapter = use_segmented_adapter
        self.use_segmented_modulator = use_segmented_modulator
        self.int_seed = int_seed
        self.model = None
    
    def _context_dimension(self) -> int:
        return self.env.encoder.context_dimension
    
    
    def _get_params(self):
        params = copy.deepcopy(self.specific_kwargs.get("policy_params", {}))
        params["device"] = self.device
        params["context_encoder"] = self.problem.context_encoder
        if "hypernetwork_kwargs" not in params:
            params["hypernetwork_kwargs"] = {}

        return params
    
    def _get_learn_function(self):
        return train_sac
    
    def _get_learn_kwargs(self):
        return {}
    
    def train(self, num_steps: int) -> Metrics:
        concat_wrapper = ConcatenateContextAndStateWrapperEnv(self.problem.get_wrapped_env(), self._context_dimension())
        #assert self.model is None
        params = self._get_params()
        kwargs = {
            'exp_conf': self.exp_conf,
            'context_dim': self._context_dimension(),
            'adapter_kwargs': params['hypernetwork_kwargs'] | {'context_size': self._context_dimension()},
        }
        if 'net_arch' in params:
            kwargs['net_arch'] = params['net_arch']
        critic_kwargs = self._get_critic_kwargs(copy.copy(kwargs))
        actor_kwargs  = self._get_actor_kwargs (copy.copy(kwargs))
            
        F = self._get_learn_function()
        self.model = F(num_steps, self.exp_conf, concat_wrapper, problem=self.problem, device=self.device, int_seed=self.int_seed, ACTOR_CLASS=self._get_actor_class(), CRITIC_CLASS=self._get_critic_class(), actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs, model_starting_point=self.model, checkpoint_path=self.specific_kwargs['log_dir'], **self._get_learn_kwargs())
    
        maybe_save_cleanrl_model(self.model, self.specific_kwargs)
        return {}
    
    def evaluate(self, number_of_episodes_total: int, test_problem: Problem) -> EpisodeRewards:
        concat_wrapper = ConcatenateContextAndStateWrapperEnv(test_problem.get_wrapped_env(), self._context_dimension())
        assert self.model is not None, "run .train() first before evaluating."
        return eval_sac(self.model, concat_wrapper, number_of_episodes_total, self.exp_conf)
    
    
    def __repr__(self) -> str:
        s = f"do_pretraining={self.do_pretraining}, pretrain_steps={self.pretrain_steps}, freeze_adapter_during_pretraining={self.freeze_adapter_during_pretraining}, freeze_main_net_initial_during_fine_tune={self.freeze_main_net_initial_during_fine_tune}, freeze_main_net_final_during_fine_tune={self.freeze_main_net_final_during_fine_tune}"
        return f"CleanSACAdapterV1({s})"
    
    def load_method_from_file(self, filename: str):
        self.model = torch.load(filename, map_location=self.device)
        self.model['device'] = self.device
        
    def _get_actor_class(self):
        return ActorAdapterOnlyMean
    
    def _get_critic_class(self):
        return SoftQNetworkAdapter
    
    def _get_actor_kwargs(self, kwargs): 
        kwargs['actor_act_at_end'] = self.exp_conf("mymethod/args/actor_act_at_end", False)
        
        if self.exp_conf("mymethod/args/actor_adapter_have_at_end", None) is not None:
            kwargs['adapter_kwargs'] = copy.deepcopy(kwargs['adapter_kwargs'])
            kwargs['adapter_kwargs']['put_adapter_before_last_layer'] = self.exp_conf("mymethod/args/put_adapter_before_last_layer", False)
            kwargs['adapter_kwargs']['put_adapter_at_end'] = True
        
        if self.exp_conf("mymethod/args/log_std_have_activation_before", None) is not None:
            kwargs['log_std_have_activation_before'] = True
            
            
        return kwargs

    def _get_critic_kwargs(self, kwargs): return kwargs
