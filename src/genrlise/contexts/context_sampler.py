from typing import List, Union
import numpy as np
from genrlise.utils.types import Context, State
from numpy.random import default_rng


class ContextSampler:
    """This is a class that can be used to sample contexts from, e.g. in a sequence, randomly from some range, etc.
    """
    def __init__(self, dims: int, **kwargs) -> None:
        self.contexts = []
        self.np_random = None
        self.dims = dims
        self.can_sample_state = False
        self._recent_episode_length = 1
    
    def set_recent_episode_length(self, ep_len: int):
        self._recent_episode_length = ep_len
    
    def seed_context_sampler(self, seed: np.random.SeedSequence):
        self.np_random = default_rng(seed)

    def sample_context(self) -> Context:
        """This samples a context and returns it. This is the main way users of this class should use it.

        Returns:
            Context: The sampled context.
        """
        ctx = self._sample_context()
        self.contexts.append(np.copy(ctx))
        return ctx
    
    def _sample_context(self) -> Context:
        """This should be implemented by the subclasses, return a context
        Returns:
            Context:
        """
        raise NotImplementedError()

    def get_number_of_contexts(self) -> int:
        """This should also be implemented by the subclasses, return an integer representing the number of available contexts, or e.g. something like float('inf') if this samples uniformly.

        Returns:
            int: How many contexts does this have.
        """
        raise NotImplementedError()
    
    def rand(self, *args, **kwargs) -> float:
        if self.np_random is not None:
            return self.np_random.random(*args, **kwargs)
        raise Exception("self.np_random is None and a random number was asked for, please call self.seed_context_sampler(seed) first")

    def sample_state(self) -> State:
        raise NotImplementedError()
    
    def reset_to_start(self):
        # Resets the state to its start
        pass
    
class SequenceContextSampler(ContextSampler):
    """This class samples from a list, in sequence, wrapping when necessary
    """
    def __init__(self, list_of_contexts: List[Context], **kwargs):
        super().__init__(**kwargs)
        list_of_contexts = [np.array(a) for a in list_of_contexts]
        self.list_of_contexts = list_of_contexts
        self.count = 0
        self.len = len(list_of_contexts)
        
    def _sample_context(self) -> Context:
        ans = np.copy(self.list_of_contexts[self.count])
        self.count = (self.count + 1) % self.len
        return ans

    def get_number_of_contexts(self) -> int:
        return self.len
    
    def reset_to_start(self):
        self.count = 0


class StateAndContextSampler(ContextSampler):
    def __init__(self, dims: int,
                 state_sampler: ContextSampler,
                 context_sampler: ContextSampler,
                 how_many_times_run_same_context: int,
                 **kwargs) -> None:
        """This samples both states and contexts

        Args:
            dims (int): dimension of context
            state_sampler (ContextSampler): the state sampler
            context_sampler (ContextSampler): the context sampler
        """
        super().__init__(dims, **kwargs)
        self.can_sample_state = True
        self.state_sampler = state_sampler
        self.context_sampler = context_sampler
        self._curr_context = self.context_sampler._sample_context()
        self.count = 0
        self.overall_step = 0
        self.how_many_times_run_same_context = how_many_times_run_same_context

    def _sample_context(self) -> Context:
        if self.count < self.how_many_times_run_same_context:
            self.count += 1
            return self._curr_context
        else:
            self.count = 1
            self.state_sampler.reset_to_start()
            self._curr_context = self.context_sampler._sample_context()
            return self._curr_context

    def sample_state(self) -> State:
        return self.state_sampler.sample_context()
    
class RepeatedContextSampler(ContextSampler):
    def __init__(self, dims: int,
                 context_sampler: ContextSampler,
                 how_many_times_run_same_context: int,
                 **kwargs) -> None:
        """This samples both states and contexts

        Args:
            dims (int): dimension of context
            state_sampler (ContextSampler): the state sampler
            context_sampler (ContextSampler): the context sampler
        """
        super().__init__(dims, **kwargs)
        self.can_sample_state = False
        self.context_sampler = context_sampler
        self._curr_context = self.context_sampler._sample_context()
        self.count = 0
        self.overall_step = 0
        self.how_many_times_run_same_context = how_many_times_run_same_context

    def _sample_context(self) -> Context:
        if self.count < self.how_many_times_run_same_context:
            self.count += 1
            return self._curr_context
        else:
            self.count = 1
            self._curr_context = self.context_sampler._sample_context()
            return self._curr_context
    
    
class LinspaceContextSampler(ContextSampler):
    """
        This samples using a linspace convention, i.e. starting from min and increasing to a max, both inclusive
    """
    def __init__(self, dims: int, mins: np.ndarray, maxs: np.ndarray, episodes: int, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        self.mins = np.array(mins).astype(np.float64)
        self.maxs = np.array(maxs).astype(np.float64)
        self.episodes = episodes
        self.curr = np.copy(self.mins)
        self.increment = (self.maxs - self.mins) / max(1, episodes - 1)
        self.counter = 0
    
    def _sample_context(self) -> Context:
        ans = np.copy(self.curr)
        self.curr += self.increment
        self.counter += 1
        eps = 1e-3
        assert np.all(np.logical_and(self.mins - eps <= ans, ans <= self.maxs + eps + self.increment)), f"{self.mins} {ans} {self.maxs} {self.counter}"
        if self.counter > self.episodes + 1:
            raise Exception("Too many resets {self.counter} -- {self.episodes}")
        return ans

    def reset_to_start(self):
        self.curr = np.copy(self.mins)
        self.counter = 0
        
    def get_number_of_contexts(self) -> int:
        return self.episodes
    
    
class IteratedContextSampler(ContextSampler):
    def  __init__(self, dims: int, samplers_to_use: List[ContextSampler], steps_for_each_sampler: Union[int, List[int]],
                  **kwargs) -> None:
        """This is a context sampler that iterates over a list of context samplers, sequentially using each to sample.

        Args:
            dims (int): _description_
            samplers (List[ContextSampler]): The samplers to use
            steps_for_each_sampler (Union[int, List[int]]): How long to use each for. If an integer, uses all samplers for the same number of steps.
        """
        super().__init__(dims, **kwargs)
        self.samplers = samplers_to_use
        if type(steps_for_each_sampler) == int:
            steps_for_each_sampler = [steps_for_each_sampler for _ in range(len(samplers_to_use))]
        assert len(steps_for_each_sampler) == len(samplers_to_use), f"Incorrect: {len(steps_for_each_sampler)} != {len(samplers_to_use)}"
        self.steps_for_each_sampler = steps_for_each_sampler
        self.sampler_index = 0
        self.num_times_sampled_current_context = 0
        self.can_sample_state = all([s.can_sample_state for s in samplers_to_use])
        self.ncalls = 0 
    
    def sample_context(self) -> Context:
        self.ncalls += 1
        assert self.sampler_index < len(self.samplers)
        self.num_times_sampled_current_context += self._recent_episode_length
        if self.num_times_sampled_current_context >= self.steps_for_each_sampler[self.sampler_index]:
            self.sampler_index += 1
            self.num_times_sampled_current_context = 0
            self.sampler_index %= len(self.samplers)
        ans = self.samplers[self.sampler_index].sample_context()
        return ans
    
    def ignore_last_one(self):
        self.num_times_sampled_current_context -= self._recent_episode_length
        if self.num_times_sampled_current_context < 0:
            self.sampler_index -= 1
            self.sampler_index %= len(self.samplers)
            self.num_times_sampled_current_context = self.steps_for_each_sampler[self.sampler_index] - self._recent_episode_length
        pass
    
    def sample_state(self) -> State:
        ans = self.samplers[self.sampler_index].sample_state()
        return ans
        
    def reset_to_start(self):
        for s in self.samplers:
            s.reset_to_start()
        self.sampler_index = 0
        self.num_times_sampled_current_context = 0

    def get_number_of_contexts(self) -> int:
        return sum([s.get_number_of_contexts() for s in self.samplers])