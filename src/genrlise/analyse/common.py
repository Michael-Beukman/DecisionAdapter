
import copy

import numpy as np
from genrlise.rlanalyse.common.result import MeanStandardResult, Result


def get_context_dimension_that_changes(result: MeanStandardResult, key: str, return_if_multiple=False) -> int:
    """This takes in the result, and returns an integer representing the context dimension that changes during evaluation. If there are multiple, this returns an error

    Args:
        result (MeanStandardResult): The result to analyse
        key (str): The key of the result to use, i.e. which set of evaluation results to consider.

    Returns:
        int: The dimension in the context that does change
    """
    
    ctxs = result.evaluation_metrics[key].contexts[0]
    those_that_change = []
    for dim in range(ctxs.shape[-1]):
        if len(np.unique(ctxs[:, dim])) > 1: 
            those_that_change.append(dim)
    
    if len(those_that_change) > 1:
        if return_if_multiple: return those_that_change
        print(f"The number of dimensions that change is more than 1: {those_that_change}")
        return those_that_change[0]
    if len(those_that_change) == 0: return 0
    return those_that_change[0]


def extract_specific_context_dimensions_from_result(result: Result, dim: int = None, return_if_multiple=False) -> Result:
    """Takes in a result and returns a modified one  where only the given context dimension is selected.

    Args:
        result (Result): 
        dim (int, optional): The dimension to select, if None, then this is set as `get_context_dimension_that_changes`. Defaults to None.

    Returns:
        Result: 
    """
    result = copy.deepcopy(result)
    for key in result.evaluation_metrics.keys():
        if 'train' in key.lower():continue
        new_dim = get_context_dimension_that_changes(result, key=key, return_if_multiple=return_if_multiple) if dim is None else dim
        try:
            result.evaluation_metrics[key].contexts = result.evaluation_metrics[key].contexts[:, :, new_dim]
        except Exception as e:
            pass
    return result
    