"""This file is focused on the genrlise library, and it uses functions largely from there to run a reproducible and clear experiment.
"""

from functools import partial
import glob
import os
import shutil
import fire
import torch
from genrlise.common.utils import set_seed
from genrlise.analyse.utils import get_all_directory_seeds, get_latest_parent_from_genrl_file
from genrlise.common.experiments.eval_utils import evaluate_model_at_the_end_of_training
from genrlise.common.experiments.experiment_utils import (
    get_context_encoder_from_config,
    get_context_sampler_from_config,
    get_environment_from_config,
    get_method_from_config,
)
from genrlise.common.infra.genrl_config import GenRLExperimentConfig, GenRLSingleRunConfig
from numpy.random import SeedSequence
from genrlise.common.path_utils import path
from genrlise.common.utils import save_compressed_pickle
from genrlise.common.vars import MODELS_DIR, NUM_CORES
from timeit import default_timer as tmr
from genrlise.contexts.problem import Problem
from genrlise.envs.wrappers.monitor_env import MonitorEnv
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList
from torch import nn

from genrlise.rlanalyse.common.common import get_full_yaml_path_from_short_name


def _log(do_log, run, *args, **kwargs):
    if not do_log:
        return
    return run.log(*args, **kwargs)


def get_specific_kwargs(do_log, DIR, exp_conf: GenRLExperimentConfig):
    log_dir = path(DIR, "logs")

    if do_log:
        callback = WandbCallback(gradient_save_freq=0, model_save_path=DIR, verbose=2, model_save_freq=50)
    else:
        callback = None
    # More callbacks
    if exp_conf("method/sb3_init_kwargs", None) is not None or exp_conf("method/sb3_learn_kwargs", None) is not None:
        assert False, "BAD STRUCTURE"
    kwargs = exp_conf("method/args", {})
    ans = (
        dict(
            sb3_callback=CallbackList(([] if callback is None else [callback])),
            sb3_init_kwargs=exp_conf("method/args/sb3_init_kwargs", {}),
            sb3_learn_kwargs=(exp_conf("method/args/sb3_learn_kwargs", {})),
            log_dir=log_dir,
        )
        | kwargs
    )
    if 'reset_num_timesteps' not in ans['sb3_learn_kwargs']:
        ans['sb3_learn_kwargs']['reset_num_timesteps'] = False
        
    print("Learn kwargs", ans['sb3_learn_kwargs'])

    for k in ["policy_kwargs", "policy_params"]:
        t = ans["sb3_init_kwargs"] if k == "policy_kwargs" else ans
        if k in t:
            if "activation_fn" in t[k]:
                name = (v := t[k])["activation_fn"].lower()
                if name == "relu":
                    act = nn.ReLU
                elif name == "tanh":
                    act = nn.Tanh
                else:
                    raise Exception(f"{name} is invalid for an activation function")
                v["activation_fn"] = act
                print(f"Activation name = {name}. class = {act}")
    return ans


def get_checkpoint_to_load(experiment_conf, int_seed):
    checkpoint = experiment_conf("method/load_checkpoint", None)
    checkpoint_seed = experiment_conf("method/load_checkpoint_if_big_seed", None)
    
    if checkpoint is None: return None
    else:
        if checkpoint_seed is not None:
            if int_seed >= 8: return checkpoint_seed
    return checkpoint

def get_seed_to_read_in(experiment_conf, int_seed):
    seed_to_read_in = experiment_conf("method/load_checkpoint_seed", int_seed)
    if seed_to_read_in >= 8:
        seed_to_read_in -= 8
    return seed_to_read_in


def main(is_local: bool, date: str, yaml_config_file: str, overall_name: str, particular_seed: int, do_log=True):
    """This uses functions and methods from the genrlise library to create a method, train it (in parallel using multiple seeds) and then to evaluate it.

    Args:
        is_local (bool): _description_
        date (str): _description_
        yaml_config_file (str): _description_
        overall_name (str): _description_
        do_log (bool, optional): _description_. Defaults to True.
    """
    experiment_conf = GenRLExperimentConfig(yaml_config_file)

    gpus = experiment_conf("infra/gpus", 0)
    if gpus == "auto":
        gpus = torch.cuda.device_count()
    CORES = experiment_conf("infra/cores", NUM_CORES(is_local))
    # ray.init(num_cpus=CORES, num_gpus=gpus)
    NUM_SEEDS = experiment_conf("infra/seeds")
    SEED_START_VAL = experiment_conf('infra/start_seed', 0)
    
    if experiment_conf('infra/disable_fp32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("Set FP32 as False")

    # @ray.remote(num_gpus=gpus/CORES)
    def single_run(numpy_seed: SeedSequence, int_seed: int):
        all_my_seeds = numpy_seed.spawn(5)
        set_seed(int_seed)

        print(f"Running with seed = {int_seed}")
        config = GenRLSingleRunConfig(experiment_conf, seed=int_seed, date=date)
        name_to_save = f"{config.unique_name()}"
        DIR = path(f"{MODELS_DIR}/{config.experiment_name}/{overall_name}/{name_to_save}")

        if int_seed == SEED_START_VAL:
            print("Running Group: ", config.hash())
            shutil.copyfile(
                yaml_config_file,
                path(f"{MODELS_DIR}/{config.experiment_name}/{overall_name}", yaml_config_file.split("/")[-1]),
            )

        if do_log:
            run = config.init_wandb(True, False)
        else:
            run = None

        LOG = partial(_log, do_log, run)

        start_time = tmr()
        og_env = get_environment_from_config(experiment_conf, seed=all_my_seeds[0], int_seed=int_seed)

        sampler = get_context_sampler_from_config(
            og_env, experiment_conf("train/context"), all_my_seeds[1], experiment_conf, int_seed
        )
        encoder = get_context_encoder_from_config(
            og_env, experiment_conf("train/context_encoder"), all_my_seeds[1], experiment_conf, int_seed
        )

        problem_to_use = Problem(og_env, sampler, encoder, do_save_all_transitions=experiment_conf("train/save_all_transitions", False))

        specific_kwargs = get_specific_kwargs(do_log, DIR, experiment_conf)
        model = get_method_from_config(og_env, experiment_conf, specific_kwargs=specific_kwargs)(
            problem_to_use, numpy_seed, int_seed
        )

        if (checkpoint_to_load := get_checkpoint_to_load(experiment_conf, int_seed)) is not None:
            # Now get the proper file. This `checkpoint_to_load` is e.g. v0210-a
            parent = get_latest_parent_from_genrl_file(get_full_yaml_path_from_short_name(checkpoint_to_load))

            seed_to_read_in = get_seed_to_read_in(experiment_conf, int_seed)
            model_file = os.path.join(get_all_directory_seeds(parent)[seed_to_read_in], 'logs','model_dir')
            print(f"Seed {int_seed} is loading in file {model_file} With loaded checkpoint code: {checkpoint_to_load} and seed to read in {seed_to_read_in}")

            model.load_method_from_file(model_file)

        if experiment_conf("train/do_train", True):
            train_metrics = model.train(experiment_conf("train/steps"))
        elif experiment_conf("train/do_probe_finetune", False):
            train_metrics = model.probe_finetune()
        else:
            raise Exception("Invalid options")
        end_time = tmr()

        env_to_use: MonitorEnv = problem_to_use.get_wrapped_env()
        if hasattr(problem_to_use, 'list_of_envs'):
            dic_to_save = {
                "all_contexts": [],
                "all_init_states": [],
                "all_episode_rewards": [],
                "all_wrapped_contexts":[],
                "all_wrapped_init_states": [],
                'all_transitions': [],
                'all_env_indices': [],
            }
            for _idx, e in enumerate(problem_to_use.list_of_envs):
                dic_to_save["all_contexts"]             .extend(e.all_contexts[:-1])
                dic_to_save["all_init_states"]          .extend(e.all_init_states[:-1])
                dic_to_save["all_episode_rewards"]      .extend(e.all_episode_rewards[1:])
                dic_to_save["all_wrapped_contexts"]     .extend(e.all_wrapped_contexts[:-1])
                dic_to_save["all_wrapped_init_states"]  .extend(e.all_wrapped_init_states[:-1])
                dic_to_save['all_transitions']          .extend(e.all_transitions)
                dic_to_save['all_env_indices'].extend([_idx for _ in e.all_contexts[:-1]])
                
        else:
            # here, we should get: Train rewards, train contexts, train states.
            dic_to_save = {
                "all_contexts": env_to_use.all_contexts[:-1],
                "all_init_states": env_to_use.all_init_states[:-1],
                "all_episode_rewards": env_to_use.all_episode_rewards[1:],
                "all_wrapped_contexts": env_to_use.all_wrapped_contexts[:-1],
                "all_wrapped_init_states": env_to_use.all_wrapped_init_states[:-1],
                'all_transitions': env_to_use.all_transitions
            }
        save_compressed_pickle(path(DIR, "train_results", mk=False), dic_to_save)
        if not experiment_conf('train/skip_main_eval', False):
            eval_rewards, start_time_eval, end_time_eval, eval_metrics = evaluate_model_at_the_end_of_training(
                model, experiment_conf, int_seed, all_my_seeds, LOG, verbose=True
            )

            save_compressed_pickle(path(DIR, "eval_results", mk=False), eval_rewards)
        else:
            eval_rewards, start_time_eval, end_time_eval, eval_metrics = None, 0, 0, None

        time_dic = {"train_time": end_time - start_time, "eval_time": end_time_eval - start_time_eval}
        LOG(time_dic)
        time_dic["train_extra"] = train_metrics
        time_dic["eval_extra"] = eval_metrics
        save_compressed_pickle(path(DIR, "assorted", mk=False), time_dic)
        
        
        if experiment_conf("train/checkpoint_frequency", -1) != -1:
            checkpoints_to_eval = experiment_conf("train/checkpoints_to_eval", None)
            print("Evaluating each checkpoint")
            if checkpoint_to_load is not None and experiment_conf("train/steps") == 0:
                # get checkpoints from previous run.
                parent = get_latest_parent_from_genrl_file(get_full_yaml_path_from_short_name(checkpoint_to_load))
                seed_to_read_in = get_seed_to_read_in(experiment_conf, int_seed)
                dir_of_checkpoints = os.path.join(get_all_directory_seeds(parent)[seed_to_read_in], 'logs','checkpoints')
            else:
                dir_of_checkpoints = os.path.join(specific_kwargs['log_dir'], 'checkpoints')
            if os.path.exists(dir_of_checkpoints):
                for model_dir in glob.glob(os.path.join(dir_of_checkpoints, '*')):
                    checkpoint_number = model_dir.split('/')[-1]
                    if checkpoints_to_eval is None or int(checkpoint_number) in checkpoints_to_eval:
                        
                        filename = os.path.join(model_dir, 'model_dir')
                        model.load_method_from_file(filename)
                        eval_rewards_checkpoint, _, _, _ = evaluate_model_at_the_end_of_training(
                            model, experiment_conf, int_seed, all_my_seeds, LOG, verbose=True
                        )
                        save_compressed_pickle(path(path(DIR, 'checkpoints', str(checkpoint_number)), "eval_results", mk=False), eval_rewards_checkpoint)
                    else:
                        print(f"Not evaluating {checkpoint_number} due to {checkpoints_to_eval=}")

            else:
                print(f"{dir_of_checkpoints} does not exist")
            
            
        if do_log:
            print("Completed")
            run.finish()

    ss = SeedSequence(12345)
    seeds = ss.spawn(NUM_SEEDS)
    assert len(seeds) == NUM_SEEDS

    single_run(seeds[particular_seed], particular_seed)

if __name__ == "__main__":
    fire.Fire(main)

