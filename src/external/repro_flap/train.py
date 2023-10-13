import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from genrlise.methods.clean.sac.networks.common import get_feedforward_model

# The FLAP models
KK = 105


class FLAPSoftQNetwork(nn.Module):
    def __init__(self, env, n_tasks: int,
                 net_arch=[KK, KK, KK]):
        super().__init__()
        self.net_arch = net_arch
        ins = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
        
        self.model = get_feedforward_model(ins, net_arch[-1], layers=net_arch[:-1], has_act_final=True)
        self.heads = nn.ModuleList([
            nn.Linear(net_arch[-1], 1) for _ in range(n_tasks)
        ])
        self.net_arch = net_arch

    def forward(self, x, a, task):
        x = torch.cat([x, a], 1)
        feats = self.model(x)
        out = self.heads[task](feats)
        return out




class FLAPActor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    def __init__(self, env, n_tasks, net_arch=[KK, KK, KK]):
        super().__init__()
        self.net_arch = net_arch
        assert len(net_arch) >= 2
        self.trunk = get_feedforward_model(np.array(env.single_observation_space.shape).prod(), net_arch[-1], layers=net_arch[:-1], has_act_final=True)
        self.fc_logstd = nn.Linear(net_arch[-1], np.prod(env.single_action_space.shape))
        self.heads = nn.ModuleList([
            nn.Linear(net_arch[-1], np.prod(env.single_action_space.shape)) for _ in range(n_tasks)
        ])
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.net_arch = net_arch

    def forward(self, x, task, force_weights=None):
        feats = self.trunk(x)
        if force_weights is not None:
            ww = force_weights[:self.heads[0].weight.numel()].reshape(self.heads[0].weight.shape)
            bb = force_weights[self.heads[0].weight.numel():]
            mean = F.linear(feats, ww, bb)
        else:
            mean = self.heads[task](feats)
        log_std = self.fc_logstd(feats)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, task, deterministic=False, force_weights=None):
        mean, log_std = self(x, task=task, force_weights=force_weights)
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

class FLAPAdapter(nn.Module):
    def __init__(self, input_space, net_arch=[2 * KK, 2 * KK, 2 * KK], output_space=2 * KK + 2) -> None:
        super().__init__()
        self.model = get_feedforward_model(input_space, output_space, layers=net_arch)
    
    def forward(self, alls):
        return self.model(alls)