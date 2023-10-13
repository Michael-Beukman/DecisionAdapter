from genrlise.common.networks.segmented_adapter_no_hypernetwork import SegmentedAdapterNoHyperNetwork
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.common.networks.segmented_adapter import SegmentedAdapterHyperNetwork

from genrlise.methods.clean.sac.networks.common import get_feedforward_model
from genrlise.methods.clean.sac.networks.myconf import NORMAL_KWARGS

# Our adapter agent's network definitions
class SoftQNetworkAdapter(nn.Module):
    def __init__(self, env, exp_conf: GenRLExperimentConfig, 
                            context_dim: int,
                            adapter_kwargs = {},
                            net_arch=[256, 256],
                            actor_act_at_end = None):
        super().__init__()
        self.context_dim = context_dim
        self.exp_conf = exp_conf
        ins = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) - self.context_dim
        adapter_kwargs['context_size'] = context_dim
        self.model = SegmentedAdapterHyperNetwork(ins, 1, layers=net_arch, **adapter_kwargs)

    def forward(self, x, a):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        x = torch.cat([x, a], 1)
        return self.model(x=x, context=ctx)


class ActorAdapterOnlyMean(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    def __init__(self, env, exp_conf: GenRLExperimentConfig, 
                            context_dim: int,
                            adapter_kwargs = {},
                            net_arch=[256, 256],
                            disable_adapter: bool = False,
                            actor_act_at_end: bool = False,
                            log_std_have_activation_before=False
                            ):
        super().__init__()
        self.actor_act_at_end = actor_act_at_end
        self.disable_adapter = disable_adapter
        self.log_std_have_activation_before = log_std_have_activation_before
        self.context_dim = context_dim
        self.exp_conf = exp_conf
        self.net_arch = net_arch
        assert len(net_arch) >= 2
        self.setup_model(env, net_arch, adapter_kwargs)
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        
    def setup_model(self, env, net_arch, adapter_kwargs):
        self.trunk = get_feedforward_model(np.array(env.single_observation_space.shape).prod() - self.context_dim, net_arch[-1], layers=net_arch[:-1], has_act_final=self.actor_act_at_end)
        
        self.fc_mean = SegmentedAdapterHyperNetwork(net_arch[-1], np.prod(env.single_action_space.shape), layers=[], **adapter_kwargs)
        self.fc_logstd = nn.Linear(net_arch[-1], np.prod(env.single_action_space.shape))
        
        if self.disable_adapter:
            self.fc_mean.enable_disable_adapter(enabled=False)
        # action rescaling
    
    def forward(self, x):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        x = self.trunk(x)
        mean = self.fc_mean(x=x, context=ctx)
        if hasattr(self, 'log_std_have_activation_before') and self.log_std_have_activation_before:
            log_std = self.fc_logstd(F.relu(x))
        else:
            log_std = self.fc_logstd(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def enable_disable_adapter(self, enabled: bool = True):
        self.disable_adapter = not enabled
        self.fc_mean.enable_disable_adapter(enabled=enabled)
    
    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std, **NORMAL_KWARGS)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


## Adapters, but the adapter module is not a hypernetwork. It takes only the context and state as a concatenated input.

class ActorAdapterOnlyMeanNotHypernetwork(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    def __init__(self, env, exp_conf: GenRLExperimentConfig, 
                            context_dim: int,
                            adapter_kwargs = {},
                            net_arch=[256, 256],
                            disable_adapter: bool = False,
                            actor_act_at_end: bool = False,
                            log_std_have_activation_before=False
                            ):
        super().__init__()
        print(f"Got a value of {disable_adapter=}. In adapter = {actor_act_at_end=} and other kwargs {adapter_kwargs=}")
        self.actor_act_at_end = actor_act_at_end
        self.disable_adapter = disable_adapter
        self.log_std_have_activation_before = log_std_have_activation_before
        self.context_dim = context_dim
        self.exp_conf = exp_conf
        self.net_arch = net_arch
        assert len(net_arch) >= 2
        self.setup_model(env, net_arch, adapter_kwargs)
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        
    def setup_model(self, env, net_arch, adapter_kwargs):
        self.trunk = get_feedforward_model(np.array(env.single_observation_space.shape).prod() - self.context_dim, net_arch[-1], layers=net_arch[:-1], has_act_final=self.actor_act_at_end)
        
        self.fc_mean = SegmentedAdapterNoHyperNetwork(net_arch[-1], np.prod(env.single_action_space.shape), layers=[], **adapter_kwargs)
        self.fc_logstd = nn.Linear(net_arch[-1], np.prod(env.single_action_space.shape))
        
        if self.disable_adapter:
            self.fc_mean.enable_disable_adapter(enabled=False)
        
    def forward(self, x):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        x = self.trunk(x)
        mean = self.fc_mean(x=x, context=ctx)
        if hasattr(self, 'log_std_have_activation_before') and self.log_std_have_activation_before:
            log_std = self.fc_logstd(F.relu(x))
        else:
            log_std = self.fc_logstd(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def enable_disable_adapter(self, enabled: bool = True):
        self.disable_adapter = not enabled
        self.fc_mean.enable_disable_adapter(enabled=enabled)
    
    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std, **NORMAL_KWARGS)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SoftQNetworkAdapterNotHypernetwork(nn.Module):
    def __init__(self, env, exp_conf: GenRLExperimentConfig, 
                            context_dim: int,
                            adapter_kwargs = {},
                            net_arch=[256, 256],
                            actor_act_at_end = None):
        super().__init__()
        self.context_dim = context_dim
        self.exp_conf = exp_conf
        ins = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) - self.context_dim
        adapter_kwargs['context_size'] = context_dim
        self.model = SegmentedAdapterNoHyperNetwork(ins, 1, layers=net_arch, **adapter_kwargs)

    def forward(self, x, a):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        x = torch.cat([x, a], 1)
        return self.model(x=x, context=ctx)

## Ablations
class ActorAdapterOnlyFeatures(ActorAdapterOnlyMean):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    def __init__(self, env, exp_conf: GenRLExperimentConfig, 
                            context_dim: int,
                            adapter_kwargs = {},
                            net_arch=[256, 256],
                            actor_act_at_end=False
                            ):
        super().__init__(env, exp_conf, context_dim, adapter_kwargs, net_arch)
        self.actor_act_at_end = actor_act_at_end
    
    def setup_model(self, env, net_arch, adapter_kwargs):
        # Here, just the trunk is adapted.
        self.trunk = SegmentedAdapterHyperNetwork(np.array(env.single_observation_space.shape).prod() - self.context_dim, net_arch[-1], layers=net_arch[:-1], **adapter_kwargs)
        self.fc_mean = nn.Linear(net_arch[-1], np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(net_arch[-1], np.prod(env.single_action_space.shape))

        
    def forward(self, x):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        x = self.trunk(x, context=ctx)
        if hasattr(self, 'actor_act_at_end') and self.actor_act_at_end:
            x = F.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


class SoftQNetworkAdapterTrunkNoActivation(nn.Module):
    def __init__(self, env, exp_conf: GenRLExperimentConfig, 
                            context_dim: int,
                            adapter_kwargs = {},
                            net_arch=[256, 256],
                            actor_act_at_end = None):
        super().__init__()
        self.context_dim = context_dim
        self.exp_conf = exp_conf
        ins = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) - self.context_dim
        adapter_kwargs['context_size'] = context_dim

        self.trunk = get_feedforward_model(ins, net_arch[-1], layers=net_arch[:-1], has_act_final=False)
        
        self.output = SegmentedAdapterHyperNetwork(net_arch[-1], 1, layers=[], **adapter_kwargs)
        
        

    def forward(self, x, a):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        x = torch.cat([x, a], 1)
        return self.output(x=self.trunk(x), context=ctx)

