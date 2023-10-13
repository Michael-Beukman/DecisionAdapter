import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from genrlise.methods.clean.sac.networks.common import get_feedforward_model
from genrlise.methods.clean.sac.networks.unaware import Actor


class BaseSACActor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    def __init__(self, env):
        super().__init__()
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
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


# cGate
class AdapterActorContextualize(BaseSACActor):
    # Like Benjamins et al. 2022 -- Contextualize
    def __init__(self, env, context_dim: int, 
                 exp_conf = None,
                 net_arch=[32, 32], adapter_net_arch = [32, 32],
                 final_policy_arch  = None,
                 final_log_std_arch = None,
                 trunk_has_act_end=False,
                 adapter_has_act_end=False
                 ):
        super().__init__(env)
        self.net_arch = net_arch
        self.context_dim = context_dim
        assert adapter_net_arch[-1] == net_arch[-1], f"BAD {adapter_net_arch} != {net_arch}"
        assert len(net_arch) >= 2
        act_shape           = np.prod(env.single_action_space.shape)
        self.trunk          = get_feedforward_model(np.array(env.single_observation_space.shape).prod() - self.context_dim, net_arch[-1], layers=net_arch[:-1], has_act_final=trunk_has_act_end)
        self.adapter_trunk  = get_feedforward_model(self.context_dim, adapter_net_arch[-1], layers=adapter_net_arch[:-1], has_act_final=adapter_has_act_end)
        
        if final_policy_arch is None: self.fc_mean   = nn.Linear(net_arch[-1],   act_shape)
        else: self.fc_mean   = get_feedforward_model(net_arch[-1],   act_shape, layers=final_policy_arch)
            
        if final_log_std_arch is None: self.fc_logstd = nn.Linear(net_arch[-1],   act_shape)
        else: self.fc_logstd = get_feedforward_model(net_arch[-1],   act_shape, layers=final_log_std_arch)
        
    def forward(self, x):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        
        state_features   = self.trunk(x)
        context_features = self.adapter_trunk(ctx)
        
        assert state_features.shape == context_features.shape
        
        new_features = state_features * context_features
        
        mean = self.fc_mean(new_features)
        log_std = self.fc_logstd(new_features)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

class AdapterCriticContextualize(nn.Module):
    def __init__(self, env, context_dim: int, 
                 exp_conf = None,
                 net_arch=[32, 32], adapter_net_arch = [32, 32],
                 final_arch=None):
        super().__init__()
        self.net_arch = net_arch
        self.context_dim = context_dim
        self.adapter_net_arch = adapter_net_arch
        ins = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) - self.context_dim
        
        self.model   = get_feedforward_model(ins, net_arch[-1], layers=net_arch[:-1], has_act_final=True)
        self.adapter = get_feedforward_model(np.prod(env.single_action_space.shape) + self.context_dim, adapter_net_arch[-1], layers=adapter_net_arch[:-1], has_act_final=True)
        
        
        if final_arch is None: self.final_val = nn.Linear(adapter_net_arch[-1], 1)
        else: self.final_val = get_feedforward_model(adapter_net_arch[-1], 1, layers=final_arch)
        

    def forward(self, x, a):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        
        state_val   = self.model(torch.cat([x, a], 1))
        context_val = self.adapter(torch.cat([ctx, a], 1))
        
        features = state_val * context_val
        
        return self.final_val(features)



# These are cGate models. HOWEVER, they each do the cgate operation (elementwise product) *at every layer*
class AdapterActorContextualizeEveryLayer(BaseSACActor):
    def __init__(self, env, context_dim: int, 
                 exp_conf = None,
                 net_arch=[32, 32], adapter_net_arch = [32, 32],
                 final_policy_arch  = None,
                 final_log_std_arch = None,
                 trunk_has_act_end=False,
                 adapter_has_act_end=False
                 ):
        super().__init__(env)
        self.net_arch = net_arch
        self.context_dim = context_dim
        assert adapter_net_arch[-1] == net_arch[-1], f"BAD {adapter_net_arch} != {net_arch}"
        assert len(net_arch) >= 2
        print(f"AdapterActorContextualizeEveryLayer actor {trunk_has_act_end=} {adapter_has_act_end=}. NET arch = {net_arch=}")
        act_shape           = np.prod(env.single_action_space.shape)
        assert len(net_arch) == 2
        
        in1 = np.array(env.single_observation_space.shape).prod() - self.context_dim
        out1 = net_arch[0]
        in2  = net_arch[0]
        out2 = net_arch[1]
        self.trunk_1          = get_feedforward_model(in1, out1, layers=[], has_act_final=True)
        self.trunk_2          = get_feedforward_model(in2, out2, layers=[], has_act_final=trunk_has_act_end)
        
        self.adapter_trunk  = get_feedforward_model(self.context_dim, adapter_net_arch[-1], layers=adapter_net_arch[:-1], has_act_final=adapter_has_act_end)
        
        if final_policy_arch is None: self.fc_mean   = nn.Linear(net_arch[-1],   act_shape)
        else: self.fc_mean   = get_feedforward_model(net_arch[-1],   act_shape, layers=final_policy_arch)
            
        if final_log_std_arch is None: self.fc_logstd = nn.Linear(net_arch[-1],   act_shape)
        else: self.fc_logstd = get_feedforward_model(net_arch[-1],   act_shape, layers=final_log_std_arch)
        
    def forward(self, x):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        
        context_features = self.adapter_trunk(ctx)
        state_1 = self.trunk_1(x)
        
        assert state_1.shape == context_features.shape
        state_1 = state_1 * context_features
        state_features = self.trunk_2(state_1)
        assert state_features.shape == context_features.shape
        
        new_features = state_features * context_features
        
        mean = self.fc_mean(new_features)
        log_std = self.fc_logstd(new_features)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

class AdapterCriticContextualizeEveryLayer(nn.Module):
    def __init__(self, env, context_dim: int, 
                 exp_conf = None,
                 net_arch=[32, 32], adapter_net_arch = [32, 32],
                 final_arch=None):
        super().__init__()
        self.net_arch = net_arch
        self.context_dim = context_dim
        self.adapter_net_arch = adapter_net_arch
        print(f"AdapterCriticContextualizeEveryLayer: NET arch = {net_arch=}")
        
        assert len(net_arch) == 2
                
        in1 = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) - self.context_dim
        out1 = net_arch[0]
        in2  = net_arch[0]
        out2 = net_arch[1]
        self.trunk_1          = get_feedforward_model(in1, out1, layers=[], has_act_final=True)
        self.trunk_2          = get_feedforward_model(in2, out2, layers=[], has_act_final=True)

        self.adapter = get_feedforward_model(np.prod(env.single_action_space.shape) + self.context_dim, adapter_net_arch[-1], layers=adapter_net_arch[:-1], has_act_final=True)
        
        
        if final_arch is None: self.final_val = nn.Linear(adapter_net_arch[-1], 1)
        else: self.final_val = get_feedforward_model(adapter_net_arch[-1], 1, layers=final_arch)
        

    def forward(self, x, a):
        DIMS = self.context_dim
        ctx = x[:, -DIMS:]
        x = x[:, :-DIMS]
        context_val = self.adapter(torch.cat([ctx, a], 1))
        
        state_1 = self.trunk_1(torch.cat([x, a], 1))
        
        assert state_1.shape == context_val.shape
        state_1 = state_1 * context_val
        state_features = self.trunk_2(state_1)
        assert state_features.shape == context_val.shape
        
        features = state_features * context_val
        
        return self.final_val(features)

