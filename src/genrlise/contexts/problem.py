import copy
from genrlise.contexts.context_encoder import ContextEncoder
from genrlise.contexts.context_sampler import ContextSampler, SequenceContextSampler, StateAndContextSampler
import genrlise as genrl
from genrlise.envs.base_env import BaseEnv
from genrlise.envs.wrappers.context_encoder_wrapper_env import ContextEncoderWrapperEnv
from genrlise.envs.wrappers.context_sampler_wrapper_env import ContextSamplerWrapperEnv
from genrlise.envs.wrappers.monitor_env import MonitorEnv


class Problem:
    """A Problem class contains an environment, a context sampler and a context encoder.
    """
    
    def __init__(self, env: genrl.BaseEnv, context_sampler: ContextSampler, context_encoder: ContextEncoder, do_save_all_transitions: bool = False, do_save_infos=False) -> None:
        self._env = env
        self.context_sampler = context_sampler
        self.context_encoder = context_encoder
        self.wrapped_env = MonitorEnv(ContextEncoderWrapperEnv(ContextSamplerWrapperEnv(self._env, self.context_sampler), self.context_encoder), do_save_all_transitions=do_save_all_transitions, do_save_infos=do_save_infos)
    
    def sample_new_context(self, set_on_env: bool = True) -> genrl.Context:
        ctx = self.context_sampler.sample_context()
        if set_on_env:
            self._env.set_context(ctx)
        return ctx
    
    def encode_context(self) -> genrl.Context:
        return self.context_encoder.get_context()
    
    def get_wrapped_env(self) -> ContextEncoderWrapperEnv:
        return self.wrapped_env
    
    def get_list_of_envs(self):
        # self.context_sampler
        new_copy_envs = [copy.deepcopy(self._env) for _ in range(100)]
        samplers = []
        if isinstance(self.context_sampler, StateAndContextSampler):
            all_states   = self.context_sampler.state_sampler.list_of_contexts
            all_contexts = self.context_sampler.context_sampler.list_of_contexts
            context_samplers = [SequenceContextSampler([c], dims=self.context_sampler.dims) for c in all_contexts]
            dims = self.context_sampler.dims
            samplers = [StateAndContextSampler(dims=dims, state_sampler=copy.deepcopy(self.context_sampler.state_sampler), context_sampler=cc, how_many_times_run_same_context=self.context_sampler.how_many_times_run_same_context) for cc in context_samplers]
        else:
            assert isinstance(self.context_sampler, SequenceContextSampler)
            all_contexts = self.context_sampler.list_of_contexts
            
            samplers = [SequenceContextSampler([c], dims=self.context_sampler.dims) for c in all_contexts]
        
        new_encoders = []
        for e in new_copy_envs:
            c = copy.deepcopy(self.context_encoder)
            c.env = e
            new_encoders.append(c)
        self.list_of_envs = [MonitorEnv(ContextEncoderWrapperEnv(ContextSamplerWrapperEnv(new, curr_sampler), enc)) for new, curr_sampler, enc in zip(new_copy_envs, samplers, new_encoders)]
        
        return self.list_of_envs
    
    def get_unwrapped_env(self) -> BaseEnv:
        return self._env
    
    def ignore_first_reset(self):
        self.wrapped_env.all_episode_rewards = self.wrapped_env.all_episode_rewards[1:]
        self.wrapped_env.all_init_states = self.wrapped_env.all_init_states[:-1]
        self.wrapped_env.all_contexts = self.wrapped_env.all_contexts[:-1]
        self.wrapped_env.all_wrapped_contexts = self.wrapped_env.all_wrapped_contexts[:-1]
        self.wrapped_env.all_wrapped_init_states = self.wrapped_env.all_wrapped_init_states[:-1]