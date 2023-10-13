import os
from typing import List


def path(*paths: List[str], mk: bool = True) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name

    Returns:
        str: 
    """
    dir = os.path.join(*paths)
    if mk:
        if '.' in (splits:=dir.split(os.sep))[-1]:
            # The final one is a file
            os.makedirs(os.path.join(*splits[:-1]), exist_ok=True)
        else:
            os.makedirs(dir, exist_ok=True)
    return dir