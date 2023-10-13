
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import numpy as np
from genrlise.analyse.utils import get_all_directory_seeds, get_latest_parent_from_genrl_file, get_single_genrlise_result, tolerant_mean
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
DATA_HOW_MANY_SEEDS = {}
__A = True
CHECK_701_SEEDS = __A
DO_SEEDS_ASSERT = __A
ASSERT_SHAPES   = __A
# These have separate runs that filled in seed number 8 - 15
RUNS_TO_HAVE_70_SEEDS = [
                        # ODE
                        'v0601', 'v0674b', 'v0673a'
                        # CP
                        'v0681', 'v0682', 'v0683', 'v0611', 
                        
                        'v0527m', 'v0527n',
                        ## Ablations
                        'v0661',
                        
                        'v0546a',
                        'v0546c',
                        
                        'v0612a', 'v0612b', 'v0612c',
                        
                        'v1619', 'v1620',
                        
                        'v1788',
                        
                        'v1607', 'v1608','v1609',
                         ]

def get_correct_v70_name(yaml_name: str):
    new_name = yaml_name.replace("v06", "v07")
    new_name = new_name.replace("v0546a", "v0746a")
    new_name = new_name.replace("v0546c", "v0746c")
    new_name = new_name.replace("v0527", "v0727")
    
    new_name = new_name.replace("v1788", "v1789")
    new_name = new_name.replace("v1619", "v1719")
    new_name = new_name.replace("v1620", "v1720")
    new_name = new_name.replace("v1607", "v1707")
    new_name = new_name.replace("v1608", "v1708")
    new_name = new_name.replace("v1609", "v1709")
    return new_name

WeightDictionary_Keys = [
    'actor.mu',
    'actor.log_std',
    'actor.latent_pi',
    'critic.q_networks[0]',
    'critic.q_networks[1]',
]
WeightDictionary = TypedDict('WeightDictionary', {
    'actor.mu': np.ndarray,
    'actor.log_std':  np.ndarray,
    'actor.latent_pi':  np.ndarray,
    'critic.q_networks[0]': np.ndarray,
    'critic.q_networks[1]': np.ndarray,
})


@dataclass
class EpisodesResults:
    """
        A run result is the result from a single execution of a model on an environment. It contains
        - rewards: List[float]
        - contexts: List[Context]
        - states: List[State]
        
        one element for each episode.
    """

    # Axes are: (seed, episode, x)
    rewards: np.ndarray   
    states: np.ndarray    
    contexts: np.ndarray  

    def __init__(self, rewards: np.ndarray, states: np.ndarray, contexts: np.ndarray, check_consistency = True, key=None, skip_asserts=False) -> None:
        self.rewards = rewards.reshape(*rewards.shape[:2])
        self.contexts = contexts
        self.states = states
        
        tt_old = self.rewards.shape[1] if len(self.rewards.shape) >= 2 else 1
        if len(self.contexts.shape) == 2: self.contexts = self.contexts[:, :, None]
        contexts = self.contexts
        if len(self.rewards.shape) >= 2 and self.rewards.shape[1] == 1504 or len(self.rewards.shape) >= 2 and self.rewards.shape[1] == 1205 :
            # this fixes the shape
            if self.rewards.shape[1] == 1205:
                self.states = np.append(self.states, self.states[:, -1:] * 0 + 1, axis=1)
            else:
                self.states = np.append(self.states, self.states[:, -1:], axis=1)
            self.rewards = np.append(self.rewards, self.rewards[:, -1:], axis=1)
            self.contexts = np.append(self.contexts, self.contexts[:, -1:], axis=1)
            rewards = self.rewards
            contexts = self.contexts
            states = self.states

        # group contexts together
        try:
            is_bad = np.unique(self.states).tolist() == [-1, -0.5, 0.5, 1] or np.unique(self.states).tolist() == [-1, -0.5, 0, 0.5, 1]
        except:
            is_bad = False
        
        if self.rewards.shape[0] > 2 or self.rewards.shape[1] == (60):
            if not is_bad and len(self.rewards.shape) >= 2:
                new_ctxs = []
                bad = False
                new_states = []
                new_rews = []
                for seed in range(len(self.rewards)):
                    cont = contexts[seed]
                    ctxs = np.unique(cont, axis=0)
                    if len(ctxs) == 1: 
                        bad = True
                    tmp_new_states = []
                    tmp_new_rews = []
                    for c in ctxs:
                        idxs = (c[None] == cont).all(axis=-1)
                        tmp_new_states.append(t:=states[seed][idxs].reshape((idxs.sum(), *states.shape[2:])).mean(axis=0))
                        tmp_new_rews.append(self.rewards[seed][idxs].reshape((idxs.sum(), 1)).mean(axis=0))
                    new_ctxs.append(ctxs)
                    new_states.append(tmp_new_states)
                    new_rews.append(tmp_new_rews)
                if not bad:
                    self.contexts = np.array(new_ctxs)
                    self.states   = np.array(new_states)
                    self.rewards  = np.array(new_rews)
        tt_new = self.rewards.shape[1] if len(self.rewards.shape) >= 2 else 1
        if not skip_asserts:
            if ASSERT_SHAPES:
                assert tt_new == tt_old // 5 or tt_new == tt_old
        self.num_seeds, self.num_episodes = len(self.rewards), len(self.rewards[0])
        if check_consistency:
            self._assert_consistent()

    def _assert_consistent(self):
        assert self.rewards.shape[:2] == (self.num_seeds, self.num_episodes), f"{self.rewards.shape} != {self.num_seeds}, {self.num_episodes}"
        assert self.contexts.shape[:1] == (
            self.num_seeds,
        ), f"{self.contexts.shape} != {self.num_seeds}, {self.num_episodes}"
        assert self.states.shape[:1] == (self.num_seeds,)  # , self.num_episodes)

    def normalise(self, optimal_result: "EpisodesResults") -> None:
        base = (optimal_result.states ** 2 * 200).reshape(*optimal_result.rewards.shape[:2])
        x = np.abs(self.rewards)
        opt = np.abs(optimal_result.rewards)
        base = np.abs(base)
        ans = 1 - (x - opt) / np.abs(base - opt)
        ans[base == opt] = 0
        assert (np.isnan(ans)).sum() == 0
        return np.clip(ans, -1e3, 1)

    def can_normalise(self, optimal_result: "EpisodesResults") -> bool:
        A = (
            self.rewards.shape == optimal_result.rewards.shape
            and self.contexts.shape == optimal_result.contexts.shape
            and self.states.shape == optimal_result.states.shape
        )
        tol = 1
        b1, b2, b3 = (
            np.all(self.rewards <= optimal_result.rewards + tol),
            np.allclose(self.contexts, optimal_result.contexts),
            np.allclose(self.states, optimal_result.states),
        )
        B = b1 and b2 and b3
        ans = A and B
        

        return ans

@dataclass
class Result:
    """
        This is a single result, consisting of training metrics, evaluation metrics, etc.
    """
    train_metrics: Optional[EpisodesResults]
    evaluation_metrics: Dict[str, EpisodesResults]
    yaml_config: GenRLExperimentConfig

    def get_baseline_result(self) -> "Result":
        """Returns a baseline result for these contexts and states.

        Returns:
            Result: The result with baseline performance for evaluation result
        """
        new_result = copy.deepcopy(self)
        if new_result.train_metrics:
            new_result.train_metrics.rewards = (new_result.train_metrics.states ** 2 * -200).reshape(
                new_result.train_metrics.rewards.shape[0], new_result.train_metrics.rewards.shape[1]
            )
            
        for k in new_result.evaluation_metrics.keys():
            new_result.evaluation_metrics[k].rewards = (new_result.evaluation_metrics[k].states ** 2 * -200).reshape(
                *new_result.evaluation_metrics[k].rewards.shape[:2]
            )
        new_result.yaml_config.set('meta/run_name', 'baseline')
        return new_result
        

    def normalise(self, optimal_results: "Result", inplace=False) -> "Result":
        """This normalises this result w.r.t. some optimal result

        Args:
            optimal_results (Result): This is the result that is optimal, with which to normalise this against.
            inplace (bool, optional): If true, sets self to the returned value. Defaults to False.

        Returns:
            Result: The normalised result
        """        
        
        new_result = Result(copy.deepcopy(self.train_metrics), copy.deepcopy(self.evaluation_metrics), copy.deepcopy(self.yaml_config))
        optimal_results = copy.deepcopy(optimal_results)
        # If the main result had more seeds than optimal, we can still normalise the eval results as they have the same contexts and states across everything
        for key in optimal_results.evaluation_metrics.keys():
            correct = new_result.evaluation_metrics[key].rewards.shape[0]
            if correct > optimal_results.evaluation_metrics[key].rewards.shape[0]:
                N = int(np.ceil(correct / optimal_results.evaluation_metrics[key].rewards.shape[0]))
                t = np.ones_like(new_result.evaluation_metrics[key].rewards.shape)
                t[0] = N
                N = tuple(t)
                optimal_results.evaluation_metrics[key].rewards  = np.tile(optimal_results.evaluation_metrics[key].rewards, N)
                optimal_results.evaluation_metrics[key].contexts = np.tile(optimal_results.evaluation_metrics[key].contexts, N)
                optimal_results.evaluation_metrics[key].states   = np.tile(optimal_results.evaluation_metrics[key].states, N)
            optimal_results.evaluation_metrics[key].rewards  = optimal_results.evaluation_metrics[key].rewards[:correct]
            optimal_results.evaluation_metrics[key].contexts = optimal_results.evaluation_metrics[key].contexts[:correct]
            optimal_results.evaluation_metrics[key].states   = optimal_results.evaluation_metrics[key].states[:correct]
        
            if (new_result.evaluation_metrics[key].rewards.shape[1]) == (correct := optimal_results.evaluation_metrics[key].rewards.shape[1]) + 1:
                new_result.evaluation_metrics[key].rewards  = new_result.evaluation_metrics[key].rewards[:correct]
                new_result.evaluation_metrics[key].contexts = new_result.evaluation_metrics[key].contexts[:correct]
                new_result.evaluation_metrics[key].states   = new_result.evaluation_metrics[key].states[:correct]
        
        if new_result.train_metrics and (new_result.train_metrics.rewards.shape[1]) == (correct := optimal_results.train_metrics.rewards.shape[1]) + 1:
            new_result.train_metrics.rewards  = new_result.train_metrics.rewards[:, :correct]
            new_result.train_metrics.contexts = new_result.train_metrics.contexts[:, :correct]
            new_result.train_metrics.states   = new_result.train_metrics.states[:, :correct]
        # Now normalise.
        assert new_result._can_normalise(optimal_results)
        # Now actually do the normalisation:
        if new_result.train_metrics:
            new_result.train_metrics.rewards = new_result.train_metrics.normalise(optimal_results.train_metrics)
        
        for key in new_result.evaluation_metrics.keys():
            new_result.evaluation_metrics[key].rewards = new_result.evaluation_metrics[key].normalise(optimal_results.evaluation_metrics[key])

        if inplace: 
            self.train_metrics = new_result.train_metrics
            self.evaluation_metrics = new_result.evaluation_metrics

        return new_result

    def sample_training_to_less_episodes(self, reduction_factor: int = 4):
        new_train_metrics = copy.deepcopy(self.train_metrics)
        episodes = self.train_metrics.num_episodes
        index = [(i + 1) * reduction_factor for i in range(episodes // reduction_factor)]
        for j in range(len(index)):
            B = index[j]
            A = max(0, B - 100)
            new_train_metrics.rewards[:, j, :] = np.mean(new_train_metrics.rewards['train_rewards'][:, A:B], axis=-1)
    
        return Result(new_train_metrics, copy.deepcopy(self.evaluation_metrics), copy.deepcopy(self.yaml_config))

    def _can_normalise(self, optimal: "Result") -> bool:
        list_a = [self.train_metrics]    + [self.evaluation_metrics[k] for k in self.evaluation_metrics]
        list_b = [optimal.train_metrics] + [optimal.evaluation_metrics[k] for k in optimal.evaluation_metrics]
        assert len(list_a) == len(list_b)
        v = True
        for a, b in zip(list_a, list_b):
            if a is not None and b is not None:
                d = a.can_normalise(b)
                v = v and d
        return v

    @staticmethod
    def load_in(yaml_name: str, parent: str = None, max_seeds: int = 1_000, include_train_data=True, read_in_kwargs: Dict[str, Any] = {}, 
                checkpoint = None, start_seeds: int = 0, skip_asserts=False, eval_keys=None) -> "Result":
        if DO_SEEDS_ASSERT: max_seeds = 16
        """This reads in the data from the yaml name, and then returns a Result instance, containing training and evaluation results.

        Args:
            yaml_name (str): The full path to the yaml file of the run under consideration.
            parent (str, optional): This overrides the parent if not None. If this is None, the parent is set to the output of `get_latest_parent_from_yaml_file`. Defaults to None.
            max_seeds (int, optional): How many seeds' worth of data to read in. Defaults to 1_000.
            include_train_data (bool, optional): If true, returns the training data too; otherwise not. Defaults to True.

        Returns:
            Result: The result of this yaml file.
        """
        exp_conf = GenRLExperimentConfig(yaml_name)
        if parent is None: parent = get_latest_parent_from_genrl_file(yaml_name)
        extra_children = []
        good = False
        if CHECK_701_SEEDS:
            for name in RUNS_TO_HAVE_70_SEEDS:
                good = good or (name in yaml_name)
            assert good or 'v08' in yaml_name or 'v09' in yaml_name or skip_asserts
            if good:
                new_name = get_correct_v70_name(yaml_name)
                assert new_name != yaml_name, f"{new_name} != {yaml_name} which is bad"
                parent2 = get_latest_parent_from_genrl_file(new_name)
                extra_children = get_all_directory_seeds(parent2)[start_seeds:max_seeds]
                assert len(set(get_all_directory_seeds(parent)[start_seeds:max_seeds] + extra_children)) == 16, f"Problem at {yaml_name} -- {len(set(get_all_directory_seeds(parent)[start_seeds:max_seeds] + extra_children))}"
        
        children = get_all_directory_seeds(parent)[start_seeds:max_seeds] + extra_children
        DATA_HOW_MANY_SEEDS[yaml_name] = len(set(children))
        output = {
            'train_rewards': [],
            'train_contexts': [],
            'train_states': [],
            'evals': {},
            'yaml_config': exp_conf
        }
        
        for child in children:
            try:
                eval_results, train_results = get_single_genrlise_result(child, checkpoint=checkpoint, **read_in_kwargs)
            except Exception as e:
                print(e)
                continue
            for k, v in eval_results.items():
                if not (eval_keys is None or k in eval_keys): continue
                if k not in (this_evals := output['evals']): this_evals[k] = {
                    'rewards': [], 'contexts': [], 'states': []
                }
                
                for key2 in this_evals[k]:
                    this_evals[k][key2].append(v[key2])
            if include_train_data:
                output['train_contexts'].append(train_results['contexts'])
                output['train_states'].append(train_results['states'])
                output['train_rewards'].append(train_results['rewards'])

        for key, alls in output['evals'].items():
            for key2, v in alls.items():
                alls[key2] = np.array(v)
                if DO_SEEDS_ASSERT and not skip_asserts: assert alls[key2].shape[0] == 16, f"{alls[key2].shape[0]} is not 16, with name {yaml_name} -- {start_seeds} {max_seeds} {checkpoint}"
                
        if include_train_data:
            for key in ['train_rewards', 'train_contexts', 'train_states',]:
                output[key] = np.array(output[key])
        
        return Result(
            train_metrics = EpisodesResults(output['train_rewards'], output['train_states'], output['train_contexts'], check_consistency=False) if include_train_data else None,
            evaluation_metrics = {key: EpisodesResults(v['rewards'], v['states'], v['contexts'], key=key, skip_asserts=skip_asserts) for key, v in output['evals'].items()},
            yaml_config = output['yaml_config']
        )
    
    def select(self, train: bool=True, eval: bool = False, context=None, state=None, contexts=None) -> "Result":
        """Selects only a subset of the contexts or states, returns a new result with only these values. Does not modify the original one.

        Args:
            train (bool, optional): Should change training data. Defaults to True.
            eval (bool, optional): Should change evaluation data. Defaults to False.
            context (_type_, optional): Filter based on context. Defaults to None.
            state (_type_, optional): Filter based on state. Defaults to None.

        Returns:
            Result: the new, filtered result.
        """
        
        new_train = copy.deepcopy(self.train_metrics)
        new_eval = copy.deepcopy(self.evaluation_metrics)
        def _select(results: EpisodesResults, context, state):
            og_c_shape = results.contexts.shape
            if results.rewards.shape[-1] == 1:
                results.rewards = results.rewards.squeeze(-1)
            good_idx = np.ones_like(results.rewards)
            if context is not None:
                good_idx = np.logical_and(good_idx, results.contexts == context)
            if state is not None:
                try:
                    good_idx = np.logical_and(good_idx, (results.states == state)[:, :, 0])
                except Exception as e:
                    good_idx = np.logical_and(good_idx, (results.states == state))
            if contexts is not None:
                temp = np.zeros_like(good_idx)
                for c in contexts:
                    temp = np.logical_or(temp, (results.contexts == c)[:, :, 0])
                good_idx = np.logical_and(good_idx, temp)
            num_seeds = len(results.rewards)
            results.rewards     = results.rewards[good_idx].reshape(num_seeds, -1)
            if len(og_c_shape) >= 3:
                results.contexts    = results.contexts[good_idx].reshape(num_seeds, -1, og_c_shape[-1])
            else:
                results.contexts    = results.contexts[good_idx].reshape(num_seeds, -1)
            results.states      = results.states[good_idx].reshape(num_seeds, -1)
            return results
        
        if train: new_train = _select(new_train, context, state)    
        if eval: new_eval = {k: _select(vv, context, state) for k, vv in new_eval.items()}
        ans = Result(new_train, new_eval, copy.deepcopy(self.yaml_config))
        if isinstance(self, MeanStandardResult):
            ans  = MeanStandardResult.from_result(ans)
        return ans

    def select_seed(self, seed: int) -> "Result":
        """Returns a copy of this result, but with only the given seed's data

        Args:
            seed (int): The seed to extract

        Returns:
            Result: 
        """
        r = copy.deepcopy(self)
        for key in r.evaluation_metrics:
            r.evaluation_metrics[key].contexts = r.evaluation_metrics[key].contexts[seed:seed+1]
            r.evaluation_metrics[key].states = r.evaluation_metrics[key].states[seed:seed+1]
            r.evaluation_metrics[key].rewards = r.evaluation_metrics[key].rewards[seed:seed+1]
        
        if self.train_metrics:
            r.train_metrics.contexts = r.train_metrics.contexts[seed:seed+1]
            r.train_metrics.states = r.train_metrics.states[seed:seed+1]
            r.train_metrics.rewards = r.train_metrics.rewards[seed:seed+1]
        return r
            
class MeanStandardResult(Result):
    train_metrics: Optional[EpisodesResults]
    evaluation_metrics: Dict[str, EpisodesResults]
    yaml_config: GenRLExperimentConfig
    """This class is effectively a 'Result', but it contains the mean, standard as its values"""
    
    @staticmethod
    def from_result(result: Result) -> "MeanStandardResult":
        
        def single_mean_std(arr: np.ndarray):
            if type(arr) == list: mean, std = tolerant_mean(arr)
            else:
                if arr.shape[-1] == 1: arr = arr.squeeze(-1)
                if len(arr.shape) == 3:
                    mean, std = np.mean(arr, axis=0, keepdims=True), np.std(arr, axis=0, keepdims=True)
                else:
                    mean, std = tolerant_mean(arr)
                    if mean.shape[0] != 1:
                        mean = mean[None]
                        std = std[None]
            new = np.vstack([mean, std])
            return new
        def mean_std(ep: EpisodesResults, kk=None):
            return EpisodesResults(
                single_mean_std(ep.rewards),
                single_mean_std(ep.states),
                single_mean_std(ep.contexts), key=kk
            )
        return MeanStandardResult(
            mean_std(result.train_metrics) if result.train_metrics is not None else None,
            {k: mean_std(evmet, kk=k) for k, evmet in result.evaluation_metrics.items()},
            result.yaml_config
            
        )