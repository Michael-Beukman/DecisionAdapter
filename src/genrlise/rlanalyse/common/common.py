import glob
import os
from typing import Any, Dict, List, Union


from common.utils import path
from genrlise.common.utils import get_full_yaml_path_from_name
from genrlise.rlanalyse.common.result import MeanStandardResult, Result


def get_yaml_filenames(parents_to_include: List[str], runs_to_include: List[str] = [], runs_to_exclude: List[str] = []) -> List[str]:
    """This takes in multiple lists and returns a list of full paths to yaml files.

    Args:
        parents_to_include (List[str]): The parents to include. Written as e.g. v0145, then includes all of the files from here
        runs_to_include (List[str], optional): Written as v0145-a, then includes this. Defaults to [].
        runs_to_exclude (List[str], optional): This is written as v0145-b and then does NOT use this run, overriding all previous settings. Defaults to [].

    Returns:
        List[str]: A list of full paths, e.g. [artifacts/config/v0145/v0145-a.yaml], etc. satisfying all of the constraints.
    """
    
    yamls = [y for p in parents_to_include for y in glob.glob(path('artifacts/config', p, '*.yaml')) if '_base' not in y]

    yamls += [get_full_yaml_path_from_name(p) for p in runs_to_include]
    runs_to_exclude = set(runs_to_exclude)
    yamls = [y for y in yamls if y.split("/")[-1].replace('.yaml', '') not in runs_to_exclude]

    return yamls

def get_full_yaml_path_from_short_name(short_yaml_name: str) -> str:
    """This takes in a short yaml name, such as v0145-o & returns a long name, such as artifacts/config/v0145/v0145-o.yaml

    Args:
        short_yaml_name (str):

    Returns:
        str:
    """
    return get_full_yaml_path_from_name(short_yaml_name)

def read_in_all_results(all_filenames: List[str], optimal_short_name: str = None, include_train_data: bool = False, max_seeds: int = 8, do_mean_standard: bool = True, do_add_baseline_results: bool = False, select_state_1: bool = True, read_in_kwargs: Dict[str, Any] = {},
                        eval_keys: List[str] = None, start_seeds = 0, checkpoint=None, skip_asserts=False) -> Union[List[Result], List[MeanStandardResult]]:
    """This reads in a list of results, and optionally normalises them

    Args:
        all_filenames (List[str]): List of full yaml filepaths, e.g. output from get_yaml_filenames()
        optimal_short_name (str, optional): Short name of the optimal run. If this is None, normalisation does not happen; otherwise it does. Defaults to None.
        include_train_data (bool, optional): Whether or not to include training data. Defaults to False.
        max_seeds (int, optional): The maximum number of seeds to read. Defaults to 8.
        do_mean_standard (bool, optional): If true, returns Mean standard results, otherwise the normal ones
        do_add_baseline_results (bool, optional): If true, adds in a baseline result, otherwise not.

    Returns:
        List[Result]: The list of results, possibly normalised
    """
    results = [Result.load_in(l, max_seeds=max_seeds, include_train_data=include_train_data, read_in_kwargs=read_in_kwargs | {'override_evals_with_assorted': False}, start_seeds=start_seeds, checkpoint=checkpoint, skip_asserts=skip_asserts, eval_keys=eval_keys) for l in all_filenames]
    if select_state_1:
        old = results
        results = [r.select(state=1, train=include_train_data, eval=True) for r in results]
        new_results = []
        KK = eval_keys[0]
        for r, o in zip(results, old):
            good_r = r
            if r.evaluation_metrics[KK].rewards.size == 0: 
                good_r = o.select(state=1/20, train=include_train_data, eval=True)
            assert good_r.evaluation_metrics[KK].rewards.size != 0
            new_results.append(good_r)
        results = new_results

    if do_add_baseline_results:
        results.append(results[0].get_baseline_result())
    
    def _rem(res: Result):
        to_remove = set(res.evaluation_metrics.keys()) - set(eval_keys)
        for r in to_remove:
            del res.evaluation_metrics[r]
    if eval_keys is not None:
        for res in results:
            _rem(res)
    
    if optimal_short_name is not None:
        optimal = Result.load_in(get_full_yaml_path_from_short_name(optimal_short_name), max_seeds=max_seeds, include_train_data=True)
        _rem(optimal)
        if select_state_1:
            optimal = optimal.select(state=1, train=True, eval=True)  
        results = [r.normalise(optimal) for r in results]
    if do_mean_standard: results = [MeanStandardResult.from_result(r) for r in results]
    
    return results
    

def any_list_item_contained_in_str(li: List[str], val: str) -> bool:
    """Checks if any of the list items are contained in val

    Args:
        li (List[str]): 
        val (str): 

    Returns:
        bool: 
    """
    for g in li:
        if g in val: return True
    return False


def get_yaml_file_from_parent(parent_dir: str) -> str:
    ans = glob.glob(os.path.join(parent_dir, '*.yaml'))
    assert len(ans) == 1, f"Incorrect L: {len(ans)} with parent dir: {parent_dir}"
    return ans[0]

