import bz2
import datetime
import glob
import hashlib
import os
import pickle
import random
from typing import Any, Dict, List, TypedDict, Union
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
import torch

from genrlise.common.path_utils import path
from genrlise.common.vars import CONFIG_DIR
from genrlise.utils.types import AgentDictionary

def get_full_yaml_path_from_name(yaml_name: str) -> str:
    """Returns the full yaml path from the name

    Args:
        yaml_name (str): The name of the yaml file, e.g. v0200-a

    Returns:
        str: The full path to the yaml file
    """
    old_fullpath = path('-'.join(yaml_name.split("-")[:-1]), yaml_name + ".yaml", mk=False)
    if os.path.exists(p:=path(CONFIG_DIR, old_fullpath, mk=False)): return p
    ans = glob.glob(path(CONFIG_DIR, '*', old_fullpath, mk=False))
    if not (good := len(ans) == 1):
        print("Incorrect length", ans, len(ans), yaml_name, old_fullpath)
    assert good
    return ans[0]
    

def torch_round(tensor: torch.Tensor, decimals: int = 2) -> torch.Tensor:
    """Rounds a specific torch tensor to x decimals

    Args:
        tensor (torch.Tensor): The tensor
        decimals (int, optional): How many to round to. Defaults to 2.

    Returns:
        torch.Tensor:
    """
    L = 10 ** decimals
    return (tensor * L).round() / L


def set_seed(seed: int = 42):
    """
        Sets the seed of numpy, random and torch
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

PICKLE_CACHE = {}
DO_CACHE = True
def load_compressed_pickle(file: str):
    if DO_CACHE:
        if file in PICKLE_CACHE: 
            print('cache len', len(PICKLE_CACHE))
            return PICKLE_CACHE[file]
    
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    if DO_CACHE:
        PICKLE_CACHE[file] = data
    
    return data

def get_dir(*paths: List[str]) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name

    Returns:
        str: 
    """
    dir = os.path.join(*paths)
    os.makedirs(dir, exist_ok=True)
    return dir

def get_date() -> str:
    """
    Returns the current date in a nice YYYY-MM-DD_H_m_s format
    Returns:
        str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def maybe_save_sb3_model(model: Union[SAC, PPO], dic: Dict[str, Any]):
    """This tries to save the model

    Args:
        model (Union[SAC, PPO]): 
        dic (Dict[str, Any]): 
    """
    if (log_dir := dic.get('log_dir', None)) is not None:
        d = path(log_dir, 'model_dir', mk=False)
        print(f"Saving model to {d}")
        model.save(d)
        
        try:
            d = path(log_dir, 'model_dir_replay_buffer', mk=False)
            print(f"Saving model to {d}")
            model.save_replay_buffer(d)
        except Exception as e:
            print('Could not save replay buffer', e)
            pass
        
def maybe_save_cleanrl_model(model: dict, dic: Dict[str, Any]):
    """This tries to save the model

    Args:
        model (Union[SAC, PPO]): 
        dic (Dict[str, Any]): 
    """
    if (log_dir := dic.get('log_dir', None)) is not None:
        d = path(log_dir, 'model_dir', mk=False)
        print(f"Saving model to {d}")
        torch.save(model, d)

def maybe_setup_sb3_logger(model: Union[SAC, PPO], dic: Dict[str, Any]):
    """Sets up this logger if 'log_dir' is found inside dic

    Args:
        model (Union[SAC, PPO]): 
        dic (Dict[str, Any]): 
    """
    if (log_dir := dic.get('log_dir', None)) is not None:
        setup_sb3_logger(model, log_dir)

def setup_sb3_logger(model: Union[SAC, PPO], log_dir: str) -> None:
    """Sets up the logger to log information to the specified log directory

    Args:
        model (Union[SAC, PPO]): 
        log_dir (str): 
    """
    new_logger = configure(log_dir, ["stdout", "csv", "json", "tensorboard"])
    model.set_logger(new_logger)
    
    
def get_submodels_from_sb3_agent(agent: SAC) -> AgentDictionary:
    return{
        'actor.mu': agent.policy.actor.mu,
        'actor.log_std': agent.policy.actor.log_std,
        'actor.latent_pi': agent.policy.actor.latent_pi,
        'critic.q_networks[0]': agent.policy.critic.q_networks[0],
        'critic.q_networks[1]': agent.policy.critic.q_networks[1],
    }
    
def get_md5sum_file(file_name: str) -> str:
    with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()