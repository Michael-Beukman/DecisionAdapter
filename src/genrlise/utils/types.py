from typing import Any, Dict, List, TypedDict, Union
import numpy as np

from genrlise.utils.metric import Metric
from enum import Enum, auto

Context = np.ndarray
State   = np.ndarray

IntAction = int
ContinuousAction = np.ndarray
Action = Union[IntAction, ContinuousAction]


Transition = np.ndarray # This is a 5 tuple (s, a, r, s')
Trajectory = np.ndarray # List[Transition]

Trajectories = List[Trajectory]


EpisodeRewards= List[List[float]] # (episodes, num_steps_per_episode)

Metrics= Dict[str, Metric]

AgentDictionary = TypedDict('AgentDictionary', {
    'actor.mu':             Any,
    'actor.log_std':        Any,
    'actor.latent_pi':      Any,
    'critic.q_networks[0]': Any,
    'critic.q_networks[1]': Any,
})


class Verbosity(Enum):
    NONE     = auto()
    PROGRESS = auto()
    DETAILED = auto()
