import bz2
from collections import defaultdict
import copy
import datetime
import glob
import hashlib
import os
import pickle
import random
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import torch
from genrlise.common.vars import SAVE_EPS, SAVE_JPG, SAVE_PDF, FIG_DPI, SAVE_PNG, SAVE_SVG

def get_date() -> str:
    """
    Returns the current date in a nice YYYY-MM-DD_H_m_s format
    Returns:
        str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def mysavefig_v2(paths: List[str], *args, **kwargs):
    """This is the same as mysavefig, but instead takes in an iterable of paths, and does the join automatically.

    Args:
        paths (List[str]): e.g. ('a, 'b', 'c', 'name.png') will result in this being saved in 'a/b/c/name.png'

    Returns:
        _type_: _description_
    """
    return mysavefig(path(*paths), *args, **kwargs)

def mysavefig(*args, **kwargs):
    """
        Saves a figure in a centralised way, also to a PDFs. Usage is just like normal matplotlib.
    """
    dirname = os.path.join(*args[0].split(os.sep)[:-1])
    os.makedirs(dirname, exist_ok=True)
    if 'dpi' not in kwargs: kwargs['dpi'] = FIG_DPI
    if 'pad_inches' not in kwargs: kwargs['pad_inches'] = 0
    if 'bbox_inches' not in kwargs: kwargs['bbox_inches'] = 'tight'
    
    
    args = list(args)
    og_name = args[0]
    if SAVE_PNG:
        plt.savefig(*args, **kwargs)
    if (SAVE_PDF or kwargs.get('save_pdf', False)):
        args[0] = og_name.split(".png")[0] + ".pdf"
        plt.savefig(*args, **kwargs)
    if SAVE_EPS:    
        args[0] = og_name.split(".png")[0] + ".eps"
        plt.savefig(*args, **kwargs)
    
    if SAVE_SVG:
        args[0] = og_name.split(".png")[0] + ".svg"
        plt.savefig(*args, **kwargs)
        
    if SAVE_JPG:
        args[0] = og_name.split(".png")[0] + ".jpg"
        plt.savefig(*args, **kwargs)



def set_seed(seed: int = 42):
    """
        Sets the seed of numpy, random and torch
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def path(p: str, *args) -> str:
    """Basically os.path.join, but the first argument can be a full path separated with os.sep
    """
    return os.path.join(*(p.split(os.sep) + list(args)))

# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_compressed_pickle(file: str):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def gpath(*paths: List[str]) -> str:
    return os.path.join(*paths)
    return dir

def get_dir(*paths: List[str]) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name

    Returns:
        str: 
    """
    dir = os.path.join(*paths)
    os.makedirs(dir, exist_ok=True)
    return dir

def get_md5sum_file(file_name: str) -> str:
    with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()
    
    
def tolerant_mean(arrs, normal_mean_std=False):
    """Calculate the mean over axis 0, but this allows the arrays to have different lengths.
    """
    if normal_mean_std: return np.mean(arrs, axis=0), np.std(arrs, axis=0)
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    mean, std = arr.mean(axis = -1), arr.std(axis=-1)
    assert np.all(mean.mask == False)
    return mean, std


def plot_mean_std(mean, std, 
                  label: str = None, ax = None):
    A = plt if ax is None else ax
    X = np.arange(len(mean))
    A.plot(X, mean, label=label)
    
    A.fill_between(X, mean - std, mean + std, alpha=0.3)
    
def get_all_directory_seeds(parent_dir: str) -> List[str]:
    """This takes in a parent directory (e.g. artifacts/models/v0102/v0102-SAC-MyCombinedHypernetwork-GroundTruth-ODE_uniform_e-1c2a58fa0ee19a9065f917b97000ec8b-all-2022-05-14_12-08-35)
    and returns a list of all of the child directories, in order.

    Args:
        parent_dir (str): 

    Returns:
        List[str]: 
    """
    
    ans = natsorted(glob.glob(os.path.join(parent_dir, '*')))
    return [a for a in ans if a[-5:] != '.yaml']

def get_yaml_file(parent_dir: str) -> str:
    ans = glob.glob(os.path.join(parent_dir, '*.yaml'))
    assert len(ans) == 1, f"Incorrect L: {len(ans)} with parent dir: {parent_dir}"
    return ans[0]

def get_single_reward_results(dir: str, return_only_evals: bool = False) -> Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]:
    """Takes in a directory DIR and returns
        - train results (rollout/ep_rew_mean) stored in DIR/logs/progress.csv
        - Dict stored in DIR/eval_reward.pbz2
        - the entire df from the above file

    Args:
        dir (str): _description_

    Returns:
        Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]: _description_
    """
    evals = load_compressed_pickle(os.path.join(dir, 'eval_reward.pbz2'))
    if return_only_evals: return evals
    progress_csv = os.path.join(dir, 'logs', 'progress.csv')
    if os.path.exists(progress_csv):
        try:
            trains = pd.read_csv(progress_csv)
            trains['id'] = np.arange(len(trains))
            t_rewards = np.array(trains['rollout/ep_rew_mean'])
        except Exception as e:
            trains = pd.DataFrame()
            t_rewards = np.array([])
    else:
        trains = load_compressed_pickle(path(dir, 'train_rewards.pbz2'))
        t_rewards = trains['reward'].sum(axis=-1)
    return t_rewards, evals, trains
    
def new_genrlise_get_single_reward_results(dir: str, return_only_evals: bool = False) -> Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]:
    """Takes in a directory DIR and returns
        - train results (rollout/ep_rew_mean) stored in DIR/logs/progress.csv
        - Dict stored in DIR/eval_reward.pbz2
        - the entire df from the above file

    Args:
        dir (str): _description_

    Returns:
        Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]: _description_
    """
    evals = load_compressed_pickle(os.path.join(dir, 'eval_results.pbz2'))
    if return_only_evals: return evals
    trains = load_compressed_pickle(os.path.join(dir, 'train_results.pbz2'))
    return evals, trains
    
    
def merge_results(all_results: List[Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]], axis: int =0) -> Dict[str, Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]]:
    """Takes a list of results as given by the function get_single_reward_results, and returns a dictionary with two keys, mean and std,
    containing the mean and std deviation.

    Args:
        all_results (List[Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]]):

    Returns:
        Dict[str, Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]]:
    """
    train_mean, train_std = tolerant_mean([r[0] for r in all_results])
    # https://stackoverflow.com/a/67584036
    df = (pd.concat([r[2] for r in all_results]))
    df_mean = df.groupby('id').agg({c: 'mean' for c in df.columns})
    df_std  = df.groupby('id').agg({c: 'std' for c in df.columns})
    
    # evals:
    eval_mean = {}
    eval_std = {}
    evals = [r[1] for r in all_results]
    for k in evals[0].keys():
        vals = []
        for e in evals:
            temp = e[k]
            if type(temp) == list:
                vals.append(temp)
            else:
                vals.append(temp['rewards'])
        eval_mean[k], eval_std[k] = np.mean(vals, axis=axis), np.std(vals, axis=axis)
        
    
    return {
        'mean': [train_mean, eval_mean, df_mean],
        'std': [train_std, eval_std, df_std]
    }
    
    
def plot_mean_std(mean: np.ndarray, std: np.ndarray, 
                  label: str = None, 
                  x: np.ndarray = None,
                  alpha: float = 0.15,
                  ax=None,
                  ignore_shade=False,
                  **kwargs):
    """Plots the mean and standard deviation in a nice and consistent way

    Args:
        mean (np.ndarray): 
        std (np.ndarray): 
        label (str, optional): . Defaults to None.
        alpha (float, optional): . Defaults to 0.15.
        ax (_type_, optional): . Defaults to None.
    """
    if x is None: x = np.arange(len(mean))
    if type(x) == list:
        x = np.array(x)
        mean = np.array(mean)
        std = np.array(std)
    ax = ax if ax else plt
    kws = copy.deepcopy(kwargs)
    l = ax.plot(x, mean, label=label, **kwargs)
    if 'linewidth' in kws: del kws['linewidth']
    if len(x.shape) == 2 and x.shape[-1] == 1:
        x = x[:, 0]
    if not ignore_shade: ax.fill_between(x, mean - std, mean + std, alpha=alpha, **kws)
    return l
    

def get_dataframe_for_barplots(mean: Dict[str, np.ndarray]):
    df = {
        'Eval. Context Range': [],
        'Eval. Reward': []
    }
    order_temp = []
    for key in mean.keys():
        df['Eval. Context Range'] += [key] * len(mean[key])
        df['Eval. Reward'] += list(mean[key])
        order_temp.append((np.mean(mean[key]), key))
    return pd.DataFrame(df), sorted(order_temp)


def convert_defaultdict_to_normal(d: dict):
    # https://localcoder.org/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_normal(v) for k, v in d.items()}
    return d


def gpath(*args) -> str:
    """This returns the path, and creates the directory if necessary

    Returns:
        str: 
    """
    dir = os.path.join(*args[:-1])
    os.makedirs(dir, exist_ok=True)
    final = os.path.join(*args)
    if '.' not in final: os.makedirs(final, exist_ok=True)
    return final