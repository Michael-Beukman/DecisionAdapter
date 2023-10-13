import math
from typing import Tuple
from matplotlib import pyplot as plt


def get_subplot_size(n_things: int, basefigsize: float = 5, force_size=None) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """Returns a decent size for a set of subplots given the number of elements to plot

    Args:
        n_things (int): _description_

    Returns:
        Tuple[int, int]: _description_
    """
    
    # a, b = min(factors, key = lambda t: t[0] + t[1])
    a = int(math.sqrt(n_things))
    b = math.ceil(n_things / a)
    if force_size is not None and force_size:
        a, b = force_size
    return (a, b), (b * basefigsize, a * basefigsize)


def mysubplots(*args, ravel=True, **kwargs):
    """
        Returns subplots, and ravels them if necessary
    """
    fig, axs = plt.subplots(*args, **kwargs)
    if hasattr(axs, '__len__'): 
        if ravel: axs = axs.ravel()
    else: axs = [axs]
    return fig, axs

def mysubplots_directly(n_things: int, basefigsize: float = 5, force_size=None, ravel=True, additional_y_size=0, *args, **kwargs):
    """Effectively combines `get_subplot_size` and `mysubplots`. Returns figure, axis

    Args:
        n_things (int): _description_
        basefigsize (float, optional): _description_. Defaults to 5.
        force_size (_type_, optional): _description_. Defaults to None.
        ravel (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    rc, s = get_subplot_size(n_things, basefigsize, force_size)
    if additional_y_size > 0:
        s = (s[0] + additional_y_size, s[1])
    return mysubplots(*rc, figsize=s, ravel=ravel, *args, **kwargs)
    
