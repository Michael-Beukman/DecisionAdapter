import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from genrlise.methods.clean.sac.networks.common import get_feedforward_model
from genrlise.methods.clean.sac.networks.myconf import NORMAL_KWARGS

# This file contains the unaware models' definitions
class SoftQNetwork(nn.Module):
    def __init__(self, env, net_arch=[256, 256], actor_act_at_end=None):
        super().__init__()
        self.net_arch = net_arch
        ins = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
        
        self.model = get_feedforward_model(ins, 1, layers=net_arch)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.model(x)




class Actor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    def __init__(self, env, net_arch=[256, 256], actor_act_at_end=False):
        super().__init__()
        self.net_arch = net_arch
        assert len(net_arch) >= 2
        self.trunk = get_feedforward_model(np.array(env.single_observation_space.shape).prod(), net_arch[-1], layers=net_arch[:-1], has_act_final=actor_act_at_end)
        
        
        self.fc_mean = nn.Linear(net_arch[-1],   np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(net_arch[-1], np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.trunk(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

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
