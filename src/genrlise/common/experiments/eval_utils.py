from timeit import default_timer as tmr
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from genrlise.common.experiments.experiment_utils import get_context_encoder_from_config, get_context_sampler_from_config, get_environment_from_config
from genrlise.common.infra.genrl_config import GenRLExperimentConfig

from genrlise.common.utils import set_seed
from genrlise.contexts.problem import Problem
from genrlise.envs.wrappers.monitor_env import MonitorEnv
from genrlise.envs.wrappers.my_sb3_monitor import MySB3Monitor
from genrlise.methods.base.genrl_method import GenRLMethod
from genrlise.utils.types import Context, EpisodeRewards, State

# This file is focused on providing utilities for running experiments.

def evaluate_model_at_the_end_of_training(model: GenRLMethod, exp_conf: GenRLExperimentConfig, int_seed: int, all_my_seeds: np.random.SeedSequence, LOG: Callable, verbose=False) -> Tuple[EpisodeRewards, float, float, List[Context], List[State], Dict[str, Any]]:
    """This function performs evaluation of a model after it has been trained. This is largely copied from v0118.py's evaluation code.

    Args:
        model (GenRLMethod): The model to evaluate
        exp_conf (GenRLExperimentConfig): The experiment config for this entire experiment
        int_seed (int): An integer seed
        all_my_seeds (np.random.SeedSequence): All of the numpy seeds
        LOG (Callable): A function that logs a dictionary, e.g. logging it to wandb
        verbose (bool, optional): Whether or not to print lots of information. Defaults to False.

    Returns:
        Tuple[EvalRewards, float, float]: _description_
    """
    print("Running Evaluation Now")
    start_time_eval = tmr()
    n_eval_episodes = exp_conf("eval/episodes")
    eval_rewards = {}
    eval_things = {}
    
    _nums = len(exp_conf("eval/context"))
    __i = 0
    extras = {}
    for little_d in exp_conf("eval/context")[1:]:
        desc = little_d['desc']
        __i+=1
        if verbose: print(f"Now in eval {__i} out of {_nums} ({desc})")
        set_seed(1_000 + int_seed)
        og_env = get_environment_from_config(exp_conf, seed=all_my_seeds[2], int_seed=int_seed, is_eval=True)
        
        
        step = 10000
        context_sampler = get_context_sampler_from_config(og_env, little_d, all_my_seeds[3], exp_conf, int_seed)
        encoder = get_context_encoder_from_config(og_env, exp_conf("train/context_encoder"), all_my_seeds[4], exp_conf, int_seed)
  
        env = MySB3Monitor(og_env)
        problem_to_use = Problem(env, context_sampler, encoder, do_save_infos=exp_conf("eval/save_lengths", False))
        
        
        model.set_problem(problem_to_use)
        extras[desc] = model.evaluate(n_eval_episodes, problem_to_use)
        
        SHOULD_REMOVE_LAST = True
        if exp_conf("meta/is_clean_rl", False):
            # Is a clean RL thing.
            problem_to_use.get_wrapped_env().reset()
        
        context_wrapper_env: MonitorEnv = problem_to_use.get_wrapped_env()
        
        
        eval_rewards[desc] = {}
        
        if SHOULD_REMOVE_LAST:
            eval_rewards[desc]['rewards'] = context_wrapper_env.all_episode_rewards[1:]
            
            eval_rewards[desc]['contexts'] = context_wrapper_env.all_contexts[:-1]
            eval_rewards[desc]['initial_states'] = context_wrapper_env.all_init_states[:-1] 
            
            eval_rewards[desc]['all_wrapped_contexts'] = context_wrapper_env.all_wrapped_contexts[:-1]
            eval_rewards[desc]['all_wrapped_init_states'] = context_wrapper_env.all_wrapped_init_states[:-1]
            
            eval_rewards[desc]['all_lengths'] = context_wrapper_env.all_episode_lengths[1:]
            eval_rewards[desc]['all_infos']   = context_wrapper_env.all_episode_infos[1:]
            
        else:
            eval_rewards[desc]['rewards'] = context_wrapper_env.all_episode_rewards
            
            eval_rewards[desc]['contexts'] = context_wrapper_env.all_contexts
            eval_rewards[desc]['initial_states'] = context_wrapper_env.all_init_states
            
            eval_rewards[desc]['all_wrapped_contexts'] = context_wrapper_env.all_wrapped_contexts
            eval_rewards[desc]['all_wrapped_init_states'] = context_wrapper_env.all_wrapped_init_states
        
        
        
        if SHOULD_REMOVE_LAST:
            all_rewards = context_wrapper_env.all_episode_rewards[1:]
        else:
            all_rewards = context_wrapper_env.all_episode_rewards
        
        assert len(all_rewards) == len(eval_rewards[desc]['contexts']), f"Incorrect {len(all_rewards)} != {len(eval_rewards[desc]['contexts'])}"

        eval_things[desc] = (np.mean(all_rewards), np.std(all_rewards))
        
        for i, r in enumerate(all_rewards):
            dic = {f"eval/{desc}/rewards": r} | {
                f"eval/{desc}/context/{j}": float(np.array([eval_rewards[desc]['contexts'][i]]).flatten()[j]) for j in range(len(eval_rewards[desc]['contexts'][i]))
            } | {
                f"eval/{desc}/state/{j}": float(np.array([eval_rewards[desc]['initial_states'][i]]).flatten()[j]) for j in range(len(eval_rewards[desc]['initial_states'][i]))
            }
            LOG(dic)
                
    end_time_eval = tmr()
    
    return eval_rewards, start_time_eval, end_time_eval, extras

