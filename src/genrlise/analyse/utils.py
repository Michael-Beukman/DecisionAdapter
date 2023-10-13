import copy
import glob
import os
from typing import Any, Dict, List, Tuple, TypedDict
from natsort import natsorted
import numpy as np
from genrlise.common.infra.genrl_config import GenRLExperimentConfig, GenRLSingleRunConfig
from genrlise.common.path_utils import path
from genrlise.common.utils import load_compressed_pickle
from genrlise.common.vars import MODELS_DIR
from genrlise.utils.types import Context, State
class _SingleResult(TypedDict):
    rewards: List[float]
    contexts: List[Context]
    states: List[State]
    


def get_latest_parent_from_genrl_file(yaml_name: str) -> str:
    """Takes in a full path to a file and returns the latest parent directory that used this config file

    Args:
        yaml_name (str): Fullpath to the config file.

    Returns:
        str: latest parent directory of the results
    """
    yaml_conf = GenRLExperimentConfig(yaml_name)
    conf = GenRLSingleRunConfig(yaml_conf, 'all', None)
    hashes = conf.hash(True, False)
    model_name = path(MODELS_DIR, yaml_conf("meta/experiment_name"), hashes, mk=False)
    candidates = sorted(glob.glob(model_name + '*'), key=lambda s: s.split(hashes + '-')[-1])
    if len(candidates) == 0:
        raise Exception(f"Model {model_name}")
    return candidates[-1]


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


def get_single_genrlise_result(dir: str, override_evals_with_assorted: bool = False, checkpoint=None, proper_key_to_use='rewards') -> Tuple[Dict[str, _SingleResult], _SingleResult]:
    """Takes in a directory DIR and returns
        - train results (rollout/ep_rew_mean) stored in DIR/logs/progress.csv
        - Dict stored in DIR/eval_reward.pbz2
        - the entire df from the above file

    Args:
        dir (str): The directory to load the results from. Must contain  `eval_results.pbz2` and `train_results.pbz2`
        override_evals_with_assorted (bool): If true, loads dir/assorted.pbz2 and replaces the eval results with that.

    Returns:
        Tuple[_SingleResult, _SingleResult]: eval, train
    """
    if checkpoint is not None:
        _evals = load_compressed_pickle(os.path.join(dir, 'checkpoints', str(checkpoint), 'eval_results.pbz2'))
    else:
        _evals = load_compressed_pickle(os.path.join(dir, 'eval_results.pbz2'))
    _trains = load_compressed_pickle(os.path.join(dir, 'train_results.pbz2'))
    
    if do_have_assorted := os.path.exists(path := os.path.join(dir, 'assorted.pbz2')):
        _assorted = load_compressed_pickle(path)
    evals = {}
    for key, val in _evals.items():
        
        if override_evals_with_assorted and do_have_assorted:
            val['rewards'] = _assorted['eval_extra'][key][proper_key_to_use]
            val['contexts'] = _assorted['eval_extra'][key]['contexts']
        
        if key == 'eval_time': continue
        
        eval_rews = val[proper_key_to_use]
        if proper_key_to_use == 'all_infos':
            eval_rews = [i[-1]['x_position'] for i in eval_rews]
        evals[key] = {
            'rewards': eval_rews,
            'states': val['initial_states'],
            'contexts': val['contexts'],
        }
    trains = {
        'rewards': _trains['all_episode_rewards'],
        'contexts': _trains['all_contexts'],
        'states': _trains['all_init_states'],
    }
    return evals, trains
    
    
def tolerant_mean(arrs, normal_mean_std=False):
    """Calculate the mean over axis 0, but this allows the arrays to have different lengths.
    """
    if normal_mean_std: return np.mean(arrs, axis=0), np.std(arrs, axis=0)
    lens = [len(i) for i in arrs]
    if len(arrs[0]) and hasattr(arrs[0][0], '__len__'):
        is_temp = True
        arr = np.ma.empty((np.max(lens), len(arrs), len(arrs[0][0]), ))
    else:
        is_temp = False
        arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        l = np.array(l)
        if l.shape[-1] == 1 and not is_temp:
            l = l.squeeze(-1)
        arr[:len(l), idx] = l
    mean, std = arr.mean(axis = 1), arr.std(axis=1)
    assert np.all(mean.mask == False)
    return mean, std
