from typing import Any, Dict
import wandb
import yaml
from genrlise.common.utils import get_md5sum_file
from genrlise.common.vars import RESULTS_DIR


def _recursive_dict_merge(a: dict, b: dict):
    # a has priority whenever something conflicts.
    all_keys = set(list(a.keys()) + list(b.keys()))
    for k in all_keys:
        if k in b and k not in a: # in b not in a
            a[k] = b[k]
        elif k in a and k not in b: # in a, not b, do nothing
            pass
        elif k in a and k in b: # in both
            assert type(a[k]) == type(b[k]) or k in ['gpus', 'cores'] and 'auto' in [a[k], b[k]], f"{k} not the same type ({type(a[k])}) != ({type(b[k])})"
            if type(a[k]) == dict:
                a[k] = _recursive_dict_merge(a[k], b[k])
            else:
                pass # otherwise, just leave in a
    return a



class GenRLExperimentConfig:
    # https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """

    def __init__(self, config_path):
        self.filename = config_path
        with open(config_path) as cf_file:
            self._data = yaml.safe_load(cf_file.read())
        
        self._check_inheritance()
        self._check_context_files()
        
    def _check_inheritance(self):
        if 'inherit' in self._data:
            inherits = self._data['inherit']
            files = inherits if type(inherits) == list else [inherits]
            files = files[::-1]  # reverse so that the last inherit file overwrites the first.
            for file in files:
                parent = GenRLExperimentConfig(file)
                # Now merge dicts
                self._data = _recursive_dict_merge(self._data, parent._data)
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        with open(filename) as cf_file:
            return yaml.safe_load(cf_file.read())
        
    
    def _check_context_files(self):
        eval_context_files = []
        if 'train' in self._data and 'context_file' in self._data['train']:
            assert 'context' not in self._data['train']
            self._data['train']['context'] = self._load_yaml_file(self._data['train']['context_file'])
            del self._data['train']['context_file']
        if 'eval' in self._data and 'context_file' in self._data['eval']:
            assert 'context' not in self._data['eval']
            eval_context_files += self._data['eval']['context_file']
            del self._data['eval']['context_file']
        
        new_ctxs = []
        for file in eval_context_files:
            new_ctxs.append(self._load_yaml_file(file))
            self._data['eval']['context'] = new_ctxs
        self._check_context_files_recursively(self._data, {}, None)
        if 'train' in self._data and 'context' in self._data['train'] and 'eval' in self._data:
            # should_add = not any(self._data['train']['context']['desc'] == e['desc'] for e in self._data['eval']['context'])
            self._data['eval']['context'] = [self._data['train']['context']] + [i for i in self._data['eval']['context'] if self._data['train']['context']['desc'] != i['desc']]

        
        # check recursively
    def _check_context_files_recursively(self, dic: dict, prev_dic: dict, prev_key):
        for key, value in dic.items():
            if type(value) == dict:
                self._check_context_files_recursively(value, dic, key)
            if key == '_files_load':
                # load multiple files
                files = value if type(value) == list else [value]
                
                new_ans = []
                for file in files:
                    new_ans.append(self._load_yaml_file(file))
                prev_dic[prev_key] = new_ans
                return
                    
    def set(self, path, value):
        vals = path.split("/")
        curr = self._data
        for v in vals[:-1]:
            curr = curr[v]
        curr[vals[-1]] = value
        
    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        recursive_dict = dict(self._data)

        if path is None:
            return recursive_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                recursive_dict = recursive_dict.get(path_item)

            value = recursive_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get(*args, **kwargs)


class GenRLSingleRunConfig:
    """
        This is a config of a single run
    """

    def __init__(self, yaml_config: GenRLExperimentConfig, seed: int, date: str) -> None:
        """A simple config class for each **run**. Effectively, this consists of a yaml_config that is shared across all runs, and a seed unique to this run.

        Args:
            yaml_config (GenRLConfig): This is the config from the yaml file
            seed (int): The seed associated with this run.
            date (str): The date of this run.
        """
        self.date = date
        self.seed = seed
        self.yaml_config = yaml_config        

        self.method = yaml_config('method/name')
        self.method_parameters = yaml_config('method')
        self.environment_name = yaml_config('env/name')

        self.project_name = yaml_config('meta/project_name')
        self.experiment_name = yaml_config('meta/experiment_name')
        self.message = yaml_config('meta/message')
        self.run_name = yaml_config('meta/run_name', f'{self.experiment_name}-{self.method}-{self.environment_name}')

        self.results_directory = yaml_config('meta/results_directory', f"{RESULTS_DIR}/{self.experiment_name}/{self.run_name}/{self.date}-{self.hash(False)}/{seed}")


    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary that can be sent to wandb that contains all the important parameters.

        Returns:
            Dict[str, Any]: [description]
        """
        return {
            'run_name': self.run_name,
            'method': self.method,
            'seed': self.seed,
            'date': self.date,
            'results_directory': self.results_directory,
            'experiment_name': self.experiment_name,
            'project_name': self.project_name,
            'method_parameters': self.method_parameters,
            'environment_name': self.environment_name,
            'message': self.message,
            'yaml': self.yaml_config._data
        }

    def hash(self, seed=False, date=True) -> str:
        """Hashes this config parameters, returning a unique identifier.

        Args:
            seed (bool, optional): Whether or not to include the seed to the hash. If yes, different seeds of the same experiment will have different hashes. Defaults to False.
            date (bool, optional): Whether or not to include the date in the hash. Defaults to True.

        Returns:
            str: The hash, with the run-name prepended.
        """
        hash = get_md5sum_file(self.yaml_config.filename)
        if seed:
            hash += "-" + str(self.seed)
        if date:
            hash += "-" + str(self.date)
        return f"{self.run_name}-" + hash

    def pretty_name(self) -> str:
        return self.hash()

    def init_wandb(self, tensorboard=True, threads=False, **kwargs) -> wandb.run:
        """
            Initialises wandb and returns the run object.
        """
        dic = dict(project=self.project_name,
                   name=self.unique_name(),
                   config=self.to_dict(),
                   job_type=self.experiment_name,
                   tags=[self.environment_name],
                   group=self.hash(),
                   sync_tensorboard=tensorboard,  # auto-upload sb3's tensorboard metrics
                   monitor_gym=True,  # auto-upload the videos of agents playing the game
                   save_code=True,
                   notes = self.yaml_config("meta/message", None))
        if threads:
            assert False
            dic['settings'] = wandb.Settings(start_method="thread")
        
        for k in kwargs:
            if k not in dic: dic[k] = kwargs[k]
        return wandb.init(**dic)

    def unique_name(self):
        return self.hash(seed=True, date=True)


def yamls_more_or_less_similar(a: GenRLExperimentConfig, b: GenRLExperimentConfig) -> bool:
    with open(a.filename, 'r') as f: dic_a = yaml.safe_load(f)
    with open(b.filename, 'r') as f: dic_b = yaml.safe_load(f)

    if 'inherit' in dic_a: del dic_a['inherit']
    if 'inherit' in dic_b: del dic_b['inherit']

    del dic_a['meta']
    del dic_b['meta']
    if (dic_a == dic_b): return True
    del dic_a['method']['policy_params']
    del dic_b['method']['policy_params']

    return dic_a == dic_b
