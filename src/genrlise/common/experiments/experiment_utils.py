"""
This file contains a collection of useful utilities for experiments.
"""

from typing import Any, Callable, Dict
from genrlise.methods.clean.sac.clean_contextualize_adapter_v12_every_layer import CleanSACContextualizeAdapterV12EveryLayer
from genrlise.methods.clean.sac.clean_sac_adapter_v1_no_hypernetwork import CleanSACAdapterV1NoHypernetwork
import numpy as np
import torch
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.contexts.context_encoder import ContextEncoder, DefaultContextEncoder, DummyDimensionGaussianContextEncoderWithMean, DummyDimensionsContextEncoder, GroundTruthContextEncoder, ConstantContextEncoder, GroundTruthNormalisedContextEncoder, NoisyConsistentContextEncoder, NoisyContextEncoder
from genrlise.contexts.context_sampler import ContextSampler, IteratedContextSampler, RepeatedContextSampler, SequenceContextSampler, StateAndContextSampler, LinspaceContextSampler
from genrlise.envs.base_cartpole_env import BaseCartPoleEnv
from genrlise.envs.base_env import BaseEnv
from genrlise.envs.cartpole import CartPoleContinuous
from genrlise.envs.complex_ode_bounded_reward import ComplexODEBoundedReward
from genrlise.methods.base.genrl_method import GenRLMethod
from genrlise.methods.clean.sac.clean_contextualize_adapter_v12 import CleanSACContextualizeAdapterV12
from genrlise.methods.clean.sac.clean_sac_adapter_v1 import CleanSACAdapterV1
from genrlise.methods.clean.sac.clean_sac_adapter_v2 import CleanSACAdapterV2
from genrlise.methods.clean.sac.clean_sac_concat import CleanSACConcat
from genrlise.methods.clean.sac.clean_sac_flap_canonical import CleanSACCanonicalFLAP
from genrlise.methods.clean.sac.clean_sac_unaware import CleanSACUnaware
from genrlise.methods.clean.sac.networks.clean_sac_adapter_v20 import CleanSACAdapterV20
from genrlise.methods.optimal_ode_model import OptimalODEModel

def get_gpu(seed_int: int, exp_conf: GenRLExperimentConfig) -> torch.device:
    """Given a seed and a config, returns a device that is either torch.cuda or cpu

    Args:
        seed_int (int): The seed, used to determine which gpu a run goes on if there are multiple GPUs
        yaml_config (GenRLExperimentConfig): The experiment config

    Returns:
        torch.device: The device
    """

    # Use multiple GPUs if available
    cores = exp_conf('infra/cores')
    if exp_conf('infra/gpus') == 'auto':
        num_gpus = torch.cuda.device_count()
        print("num gpus = ", num_gpus, 'seed_int = ', seed_int, 'cores=', cores)
        if num_gpus == 0:
            return torch.device('cpu') 
        if num_gpus == 1: return torch.device('cuda:0') 

        return torch.device(f'cuda:{0 if seed_int < cores // 2 else 1}') 
    

    if exp_conf('infra/gpus') == 1: return torch.device('cuda:0') 
    if exp_conf('infra/gpus') == 0: return torch.device('cpu') 
    if exp_conf('infra/gpus') == 2: return torch.device(f'cuda:{0 if seed_int < cores // 2 else 1}') 


def get_environment_from_config(conf: GenRLExperimentConfig, seed: np.random.SeedSequence, int_seed: int, is_eval: bool = False) -> BaseEnv:
    """Given a config, create an environment from this and seed it properly.

    Args:
        conf (GenRLExperimentConfig): Config for the experiment
        seed (np.random.SeedSequence): The seed to use
        int_seed (int): An integer seed
        is_eval (bool): If this is an evaluation setting

    Returns:
        BaseEnv: The environment
    """
    all_args = conf.get("env/args", {})
    if is_eval:
        all_args = all_args | conf.get("env/eval_args", {})
    
    env = conf("env/name")
    if 'mujoco' in env.lower():
        from genrlise.envs.mujoco.ant_v3_with_timelimit_density_v2 import MujocoAntEnvTimeLimitDensityV2
    
    mapping = {
        'ComplexODEBoundedReward':                  lambda args: ComplexODEBoundedReward(context=conf.get("env/context", [1, 1]), seed=seed, **args),
        'CartPoleContinuous':                       lambda args: CartPoleContinuous(context=conf.get("env/context", BaseCartPoleEnv.DEFAULT_CONTEXT), seed=seed, **args),
        
        'MujocoAntEnvTimeLimitDensityV2':           lambda args: MujocoAntEnvTimeLimitDensityV2(context=conf.get("env/context", MujocoAntEnvTimeLimitDensityV2.DEFAULT_CONTEXT), int_seed=int_seed, seed=seed, **args),
    }

    if int_seed is None: int_seed = seed

    
    if env not in mapping: raise Exception(f"{env} is not in the allowed list of envs: {mapping.keys()}")
    E = mapping[env](all_args)
    if env != 'DMCCheetah' and env != 'BipedalWalker': E.action_space.seed(int_seed)
    return E

def get_context_sampler_from_config(env: BaseEnv, dic: Dict[str, Any], 
                                    seed: np.random.SeedSequence, conf: GenRLExperimentConfig = None, seed_int = None) -> ContextSampler:
    """This takes in a dictionary (from the config.yaml) and returns a context sampler

    Args:
        env (BaseEnv): The base environment to use with this wrapper
        dic (Dict[str, Any]): The config dict, must have 'name' property
        seed (np.random.SeedSequence): Seed to seed the context wrapper with
        conf (GenRLExperimentConfig, optional): The experiment config. Defaults to None.
        seed_int (_type_, optional): An integer seed. Defaults to None.

    Returns:
        ContextSampler: The context sampler to use given the config
    """

    mapping = {
        'SequenceSampler': SequenceContextSampler,
        'Linspace': LinspaceContextSampler,
    }
    name = dic['name']
    if name == 'StateAndContext':
        dims = {'dims': dic['dims']}
        sampler = StateAndContextSampler(
            state_sampler=get_context_sampler_from_config(env, dims | dic['state'], seed, conf, seed_int),
            context_sampler=get_context_sampler_from_config(env, dims | dic['context'], seed, conf, seed_int),
            **dic)
    elif name == 'IteratedContextSampler':
        dims = {'dims': dic['dims']}
        sampler = IteratedContextSampler(
            samplers_to_use=[get_context_sampler_from_config(env, dims | _samp, seed, conf, seed_int) for _samp in dic['samplers']],
            **dic)
    elif name == 'RepeatedContextSampler':
        dims = {'dims': dic['dims']}
        sampler = RepeatedContextSampler(
            context_sampler=get_context_sampler_from_config(env, dims | dic['context'], seed, conf, seed_int),
            **dic)
    else:
        sampler = mapping[name](**dic)
    sampler.seed_context_sampler(seed)
    return sampler



def get_context_encoder_from_config(env: BaseEnv, dic: Dict[str, Any], 
                                    seed: np.random.SeedSequence, conf: GenRLExperimentConfig = None, seed_int = None) -> ContextEncoder:
    """This takes in a dictionary (from the config.yaml) and returns a context encoder

    Args:
        env (BaseEnv): The base environment to use with this wrapper
        dic (Dict[str, Any]): The config dict, must have 'name' property
        seed (np.random.SeedSequence): Seed to seed the context wrapper with
        conf (GenRLExperimentConfig, optional): The experiment config. Defaults to None.
        seed_int (_type_, optional): An integer seed. Defaults to None.

    Returns:
        ContextEncoder: Encoder
    """

    mapping = {
        'Identity':   GroundTruthContextEncoder,
        'Normalised': GroundTruthNormalisedContextEncoder,
        'Default': DefaultContextEncoder,
        'Constant': ConstantContextEncoder
    }
    if dic is None or dic == {}:
        dic = {'name': 'Identity'}
    name = dic['name']

    if name == 'Noisy':
        inner = get_context_encoder_from_config(env, dic['base'], seed, conf, seed_int)
        return NoisyContextEncoder(env=env, encoder=inner,
                                    device=get_gpu(seed_int, conf),
                                    seed=seed,
                                    **dic.get('args', {}))
    if name == 'NoisyConsistent':
        inner = get_context_encoder_from_config(env, dic['base'], seed, conf, seed_int)
        return NoisyConsistentContextEncoder(env=env, encoder=inner,
                                    device=get_gpu(seed_int, conf),
                                    seed=seed,
                                    **dic.get('args', {}))
    if name == 'DummyDimensions':
        inner = get_context_encoder_from_config(env, dic['base'], seed, conf, seed_int)
        return DummyDimensionsContextEncoder(env=env, encoder=inner,
                                    device=get_gpu(seed_int, conf),
                                    seed=seed,
                                    **dic.get('args', {}))
    if name == 'DummyDimensionGaussian':
        inner = get_context_encoder_from_config(env, dic['base'], seed, conf, seed_int)
        return DummyDimensionGaussianContextEncoderWithMean(env=env, encoder=inner,
                                    device=get_gpu(seed_int, conf),
                                    seed=seed,
                                    **dic.get('args', {}))
    return mapping[name](
        env=env,
        device=get_gpu(seed_int, conf),
        seed=seed,
        **dic.get('args', {}))


def get_method_from_config(env: BaseEnv, exp_conf: GenRLExperimentConfig, specific_kwargs: Dict[str, Any] = {}) -> Callable[[BaseEnv, np.random.SeedSequence, int], GenRLMethod]:
    """Takes in an environment and a config and returns a function that takes in env, seedsequence and an integer seed and returns a `GenRLMethod`

    Args:
        env (BaseEnv): The environment to run on
        exp_conf (GenRLExperimentConfig): The config for this experiment

    Returns:
        Callable[[BaseEnv, np.random.SeedSequence, int], GenRLMethod]: A function that returns a GenRLMethod
    """

    mapping = {
        'OptimalODE': lambda env, seed, seed_int: OptimalODEModel(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        
        # Unawre
        'Clean_SAC_Unaware': lambda env, seed, seed_int: CleanSACUnaware(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        # Concat
        'Clean_SAC_Concat': lambda env, seed, seed_int: CleanSACConcat(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        # Adapter -- Ours
        'Clean_SAC_Adapter_V1': lambda env, seed, seed_int: CleanSACAdapterV1(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        'Clean_SAC_Adapter_V2': lambda env, seed, seed_int: CleanSACAdapterV2(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        'Clean_SAC_Adapter_V20': lambda env, seed, seed_int: CleanSACAdapterV20(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        # cGate
        'Clean_SAC_Adapter_Contextualize_V12': lambda env, seed, seed_int: CleanSACContextualizeAdapterV12(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        # FLAP
        'Clean_SAC_Other_Canonical_FLAP': lambda env, seed, seed_int: CleanSACCanonicalFLAP(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        
        
        # Other Baselines
                
        'Clean_SAC_Adapter_Contextualize_V12_Every_Layer': lambda env, seed, seed_int: CleanSACContextualizeAdapterV12EveryLayer(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
        
        'Clean_SAC_Adapter_V1_No_Hypernetwork': lambda env, seed, seed_int: CleanSACAdapterV1NoHypernetwork(env, device=get_gpu(seed_int, exp_conf), seed=seed, exp_conf=exp_conf, int_seed=seed_int, specific_kwargs=specific_kwargs, **specific_kwargs.get('kwargs', {})),
        
    }

    model = exp_conf("method/name")
    if model not in mapping: raise Exception(f"{model} is not in the allowed list of models: {mapping.keys()}")
    return mapping[model]
