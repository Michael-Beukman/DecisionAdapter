# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Code is from CleanRL
import argparse
import copy
import os
import random
import time
from distutils.util import strtobool
from xml.etree.ElementInclude import default_loader

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.common.utils import maybe_save_cleanrl_model
from genrlise.contexts.problem import Problem
from genrlise.methods.clean.common.working_sync_vector_env import WorkingSyncVectorEnv
from genrlise.methods.clean.sac.networks.unaware import Actor, SoftQNetwork
from genrlise.methods.hyper.hyper_sac_base import MyContextReplayBuffer


def parse_args(exp_conf: GenRLExperimentConfig, int_seed):
    
    
    args = argparse.Namespace(
        exp_name="base_clean_sac",
        seed=int_seed,
        torch_deterministic=True,
        cuda=True,
        track=False,
        wandb_project_name="cleanRL",
        wandb_entity=None,
        capture_video=False,
        total_timesteps=exp_conf("train/steps"),
        buffer_size=1000000,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        exploration_noise=0.1,
        learning_starts=5000.0,
        policy_lr=0.0003,
        q_lr=0.001,
        policy_frequency=2,
        target_network_frequency=1,
        noise_clip=0.5,
        alpha=0.2,
        autotune=True,
    )
    # fmt: off
    return args


def make_env(proper_env, seed, idx, capture_video, run_name):
    assert idx == 0
    def thunk():
        env = proper_env
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if type(proper_env.seed) != np.random.bit_generator.SeedSequence:
            env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk




def eval_sac(model, env: gym.Env, n_eval_episodes: int, exp_conf: GenRLExperimentConfig):
    all_rewards = []
    device = model['device']
    clean_obs_space, norm_mode = get_clean_obs_space_and_norm_mode(env.observation_space, exp_conf)
    for ep in range(n_eval_episodes):
        obs = env.reset()
        obs = maybe_normalise_observations(obs, clean_obs_space, norm_mode)
        done = False
        tot_r = 0
        while not done:
            action, _, _ = model['model']['actor'].get_action(torch.Tensor(obs[None]).to(device), deterministic=True)
            action = action.detach().cpu().numpy()
            next_obs, reward, done, info = env.step(action[0])
            
            next_obs = maybe_normalise_observations(next_obs, clean_obs_space, norm_mode)
            
            tot_r += reward
            obs = next_obs
        all_rewards.append(tot_r)
    return np.array(all_rewards)

def get_clean_obs_space_and_norm_mode(obs_space, exp_conf):
    norm_mode = exp_conf("train/sac/observation_norm_mode", 'none')
    default_low = 0
    if norm_mode == 'negone_one': default_low = -1 # so that infinite ones do not change after normalisation.
    clean_obs_space = copy.deepcopy(obs_space)
    for i in range(len(clean_obs_space.low)):
        if np.isinf(clean_obs_space.low[i]) or clean_obs_space.low[i] <= -10**5:
            clean_obs_space.low[i] = default_low
        if np.isinf(clean_obs_space.high[i]) or clean_obs_space.high[i] >= 10**5:
            clean_obs_space.high[i] = 1
    return clean_obs_space, norm_mode
    

def maybe_normalise_observations(obs, obs_space, mode: str = 'none'):
    def inner():
        if mode == 'none': return obs
        elif mode == 'zero_one': return (obs - obs_space.low) / (obs_space.high - obs_space.low)
        elif mode == 'negone_one': return 2 * (obs - obs_space.low) / (obs_space.high - obs_space.low) - 1
        else:
            assert False, f"Mode {mode} is not supported"
    ans = inner()
    return ans

def train_sac(num_timesteps, exp_conf: GenRLExperimentConfig, environment: gym.Env, problem: Problem, device, int_seed: int,
              ACTOR_CLASS=Actor, CRITIC_CLASS=SoftQNetwork, actor_kwargs: dict = {}, critic_kwargs: dict = {}, use_context_replay_buffer: bool = False,
              model_starting_point = None, should_train_critic: bool = True,
              checkpoint_path: str = None):
    args = parse_args(exp_conf, int_seed)
    checkpoint_freq = exp_conf("train/checkpoint_frequency", -1)
    args.total_timesteps = num_timesteps
    run_name = f"{exp_conf('meta/run_name')}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    # env setup
    envs = WorkingSyncVectorEnv([make_env(environment, args.seed, 0, args.capture_video, run_name)])
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    
    if model_starting_point is not None:
        actor               = model_starting_point['model']['actor']
        qf1                 = model_starting_point['model']['qf1']
        qf2                 = model_starting_point['model']['qf2']
        qf1_target          = model_starting_point['model']['qf1_target']
        qf2_target          = model_starting_point['model']['qf2_target']
        qf1_target          = model_starting_point['model']['qf1_target']
        qf2_target          = model_starting_point['model']['qf2_target']
    else:
        actor = ACTOR_CLASS(envs, **actor_kwargs).to(device)
        qf1 = CRITIC_CLASS(envs, **critic_kwargs).to(device)
        qf2 = CRITIC_CLASS(envs, **critic_kwargs).to(device)
        qf1_target = CRITIC_CLASS(envs, **critic_kwargs).to(device)
        qf2_target = CRITIC_CLASS(envs, **critic_kwargs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
    
    
    def save_model(global_step):
        if checkpoint_freq == -1 or checkpoint_path is None: return
        elif global_step % checkpoint_freq == 0:
          m = {
                "model": {
                    "actor": actor,
                    "qf1": qf1,
                    "qf2": qf2,
                    "qf1_target": qf1_target,
                    "qf2_target": qf2_target,
                    "qf1_target": qf1_target,
                    "qf2_target": qf2_target,
                },
                'device': device
            }
          myd = os.path.join(checkpoint_path, 'checkpoints', str(global_step))
          print(f"Saving Checkpoint at step {global_step} to filename {myd}")  
          os.makedirs(myd, exist_ok=True)
          save_dic = {'log_dir': myd}
          maybe_save_cleanrl_model(m, save_dic)
        
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    
    clean_obs_space, norm_mode = get_clean_obs_space_and_norm_mode(envs.single_observation_space, exp_conf)
    
    _class_name = MyContextReplayBuffer if use_context_replay_buffer else ReplayBuffer
    _kwargs = dict(context_encoder=problem.context_encoder) if use_context_replay_buffer else {}
    rb = _class_name(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
        **_kwargs
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    obs = maybe_normalise_observations(obs, clean_obs_space, norm_mode)
    
    
    for global_step in range(args.total_timesteps):
        if global_step % 10_000 == 0:
            print(f"Step {global_step:<8} of {args.total_timesteps:<8} => {np.round(100*global_step/args.total_timesteps)}%")
        save_model(global_step)
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        next_obs = maybe_normalise_observations(next_obs, clean_obs_space, norm_mode)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
                real_next_obs[idx] = maybe_normalise_observations(real_next_obs[idx], clean_obs_space, norm_mode)
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target
                ).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            if should_train_critic:
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print(f"{global_step:<8} SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

    return {
        "model": {
            "actor": actor,
            "qf1": qf1,
            "qf2": qf2,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
        },
        'device': device
    }


if __name__ == "__main__":
    train_sac()
