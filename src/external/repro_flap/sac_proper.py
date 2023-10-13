# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# From CleanRL, altered to implement the FLAP model
import argparse
import copy
import os
import random
import time
from typing import List


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from external.repro_flap.train import FLAPActor, FLAPAdapter, FLAPSoftQNetwork

from genrlise.common.infra.genrl_config import GenRLExperimentConfig
from genrlise.common.utils import maybe_save_cleanrl_model
from genrlise.methods.clean.common.working_sync_vector_env import WorkingSyncVectorEnv


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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
        q_lr=0.0003,
        policy_frequency=2,
        target_network_frequency=1,
        noise_clip=0.5,
        alpha=0.2,
        autotune=True,
        adapter_lr=3*1e-4
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
            print(type(proper_env.seed), type(proper_env.seed) == np.random.bit_generator.SeedSequence)
            env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def eval_sac_flap_canonical(model, env: gym.Env, n_eval_episodes: int, 
             exp_conf: GenRLExperimentConfig, task=None, use_adapter=False, adaptation_steps=None,
             use_context_adapter=False):
    all_rewards = []
    device = model['device']
    clean_obs_space, norm_mode = get_clean_obs_space_and_norm_mode(env.observation_space, exp_conf)
    print("Evaluating  with num eps", n_eval_episodes, f'and {use_context_adapter=}')
    for ep in range(n_eval_episodes):
        print(f"EVAL {ep} / {n_eval_episodes}",)
        obs = env.reset()
        obs = maybe_normalise_observations(obs, clean_obs_space, norm_mode)
        done = False
        tot_r = 0
        step0 = True
        stepcount = 0
        while not done:
            stepcount += 1
            if use_adapter:
                if step0: # arbitrarily act
                    action, _, _ = model['model']['actor'].get_action(torch.Tensor(obs[None]).to(device), deterministic=True, task=0)
                else:
                    inputs = np.concatenate(list(map(lambda x: np.array([x]).flatten(), [prev_obs, action, reward, obs])))
                    if stepcount <= adaptation_steps:
                        if use_context_adapter:
                            weights = model['model']['ctx_adapter'](env.get_context()).flatten()
                        else:
                            weights = model['model']['adapter'](torch.tensor(inputs, dtype=torch.float32).to(device))
                    else:
                        pass
                    action, _, _ = model['model']['actor'].get_action(torch.Tensor(obs[None]).to(device), deterministic=True, task=0, force_weights=weights)
            else:
                action, _, _ = model['model']['actor'].get_action(torch.Tensor(obs[None]).to(device), deterministic=True, task=task)
            action = action.detach().cpu().numpy()
            next_obs, reward, done, info = env.step(action[0])
            
            next_obs = maybe_normalise_observations(next_obs, clean_obs_space, norm_mode)
            
            tot_r += reward
            prev_obs = obs
            obs = next_obs
            step0 = False
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

def train_sac_flap_proper(num_timesteps, exp_conf: GenRLExperimentConfig, environments: List[gym.Env], device, int_seed: int, context_dimension,
              ACTOR_CLASS=FLAPActor, CRITIC_CLASS=FLAPSoftQNetwork, actor_kwargs: dict = {}, critic_kwargs: dict = {}, adapter_kwargs = {},
              model_starting_point = None, should_train_critic: bool = True,
              checkpoint_path: str = None):
    ALL_LOGS = []
    args = parse_args(exp_conf, int_seed)
    checkpoint_freq = exp_conf("train/checkpoint_frequency", -1)
    args.total_timesteps = num_timesteps
    run_name = f"{exp_conf('meta/run_name')}"


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

    all_my_envs = []
    for e in environments:
        all_my_envs.append(WorkingSyncVectorEnv([make_env(e, args.seed, 0, args.capture_video, run_name)]))
        assert isinstance(all_my_envs[-1].single_action_space, gym.spaces.Box), "only continuous action space is supported"

    
    if model_starting_point is not None:
        actor               = model_starting_point['model']['actor']
        qf1                 = model_starting_point['model']['qf1']
        qf2                 = model_starting_point['model']['qf2']
        qf1_target          = model_starting_point['model']['qf1_target']
        qf2_target          = model_starting_point['model']['qf2_target']
        qf1_target          = model_starting_point['model']['qf1_target']
        qf2_target          = model_starting_point['model']['qf2_target']
    else:
        actor_kwargs['n_tasks']  = len(environments)
        critic_kwargs['n_tasks'] = len(environments)
        actor = ACTOR_CLASS(all_my_envs[0], **actor_kwargs).to(device)
        qf1 = CRITIC_CLASS(all_my_envs[0], **critic_kwargs).to(device)
        qf2 = CRITIC_CLASS(all_my_envs[0], **critic_kwargs).to(device)
        qf1_target = CRITIC_CLASS(all_my_envs[0], **critic_kwargs).to(device)
        qf2_target = CRITIC_CLASS(all_my_envs[0], **critic_kwargs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
    
    
    num_states = np.array(all_my_envs[0].single_observation_space.shape).prod()
    numacts = np.prod(all_my_envs[0].single_action_space.shape)
    
    adapter_outputs = numacts * (1 + actor.net_arch[-1])
    
    adapter = FLAPAdapter(input_space=2 * num_states + 1 + numacts, output_space=adapter_outputs, **adapter_kwargs).to(device)
    context_conditioned_adapter = FLAPAdapter(input_space=context_dimension, output_space=adapter_outputs, **adapter_kwargs).to(device)
    a = get_n_params(actor)
    b = get_n_params(qf1)
    c = get_n_params(adapter)
    d = get_n_params(context_conditioned_adapter)
    print(f"Info:: Number of Params {a:<10,} {b:<10,} {c:<10,} {d} {a+b+c:<10,}") 
    def save_model(global_step, override_test=False):
        if checkpoint_freq == -1 or checkpoint_path is None: return
        elif global_step % checkpoint_freq == 0 or global_step == 0 or override_test != False:
          m = {
                "model": {
                    "actor": actor,
                    "qf1": qf1,
                    "qf2": qf2,
                    "qf1_target": qf1_target,
                    "qf2_target": qf2_target,
                    "qf1_target": qf1_target,
                    "qf2_target": qf2_target,
                    'adapter': adapter,
                    'ctx_adapter': context_conditioned_adapter
                },
                'device': device
            }
          if override_test == False:
            myd = os.path.join(checkpoint_path, 'checkpoints', str(global_step))
          else:
            myd = os.path.join(checkpoint_path, 'checkpoints', str(global_step) + f"_test_{override_test}")
          print(f"Saving Checkpoint at step {global_step} to filename {myd}")  
          os.makedirs(myd, exist_ok=True)
          save_dic = {'log_dir': myd}
          maybe_save_cleanrl_model(m, save_dic)
        
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    adapter_optimizer = optim.Adam(list(adapter.parameters()), lr=args.adapter_lr)
    adapter_optimizer_ctx = optim.Adam(list(context_conditioned_adapter.parameters()), lr=args.adapter_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(all_my_envs[0].single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    for e in all_my_envs:
        e.single_observation_space.dtype = np.float32
    
    clean_obs_space, norm_mode = get_clean_obs_space_and_norm_mode(all_my_envs[0].single_observation_space, exp_conf)
    
    
    all_my_replay_buffers = []    
    for e in all_my_envs:
        _class_name = ReplayBuffer
        _kwargs = {}
        all_my_replay_buffers.append(_class_name(
            args.buffer_size,
            e.single_observation_space,
            e.single_action_space,
            device,
            handle_timeout_termination=True,
            **_kwargs
        ))
    start_time = time.time()

    adapter_loss_func = nn.MSELoss()
    # TRY NOT TO MODIFY: start the game
    NUM_TRAIN_STEPS_PER_ITER = 1000
    DATA_COLLECTION_EPISODES = 1000

    global_step = 0
    has_gotten_contexts = False
    TORCH_CTX = []
    save_model(global_step)
    while global_step < args.total_timesteps: # while not done in the paper
        tmp_rl = []
        for task_index in range(len(all_my_envs)): # collect data step
            obs = all_my_envs[task_index].reset()
            if not has_gotten_contexts:
                TORCH_CTX.append(all_my_envs[task_index].envs[0].get_context())
            obs = maybe_normalise_observations(obs, clean_obs_space, norm_mode)
            for _ in range(DATA_COLLECTION_EPISODES):
                global_step += 1
                if global_step % 10_000 == 0:
                    print(f"Step {global_step:<8} of {args.total_timesteps:<8} => {np.round(100*global_step/args.total_timesteps)}%")
                save_model(global_step)
                if global_step < args.learning_starts:
                    actions = np.array([all_my_envs[task_index].single_action_space.sample() for _ in range(all_my_envs[task_index].num_envs)])
                else:
                    
                    actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), task=task_index)
                    actions = actions.detach().cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, dones, infos = all_my_envs[task_index].step(actions)
                next_obs = maybe_normalise_observations(next_obs, clean_obs_space, norm_mode)
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                for info in infos:
                    if "episode" in info.keys():
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        tmp_rl.append({'return': info['episode']['r']})
                        break

                # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
                real_next_obs = next_obs.copy()
                for idx, d in enumerate(dones):
                    if d:
                        real_next_obs[idx] = infos[idx]["terminal_observation"]
                        real_next_obs[idx] = maybe_normalise_observations(real_next_obs[idx], clean_obs_space, norm_mode)
                all_my_replay_buffers[task_index].add(obs, real_next_obs, actions, rewards, dones, infos)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
        has_gotten_contexts = True
        ALL_LOGS.append(tmp_rl)
        tmp_train = []
        all_sum_qfloss = torch.tensor(0)
        all_sum_actorloss = torch.tensor(0)
        all_sum_alpha_loss = torch.tensor(0)
        all_sum_adapter_loss = torch.tensor(0)
        all_sum_adapter_ctx_loss = torch.tensor(0)
        
        if global_step >= args.total_timesteps:
            save_model(global_step, override_test='before_train_final')
        
        for train_step in range(NUM_TRAIN_STEPS_PER_ITER): # Now update this
            _r = lambda x: np.round(x, 4)
            print(f'train step {global_step:<5}/{args.total_timesteps} {train_step:<5}/{NUM_TRAIN_STEPS_PER_ITER} {_r(all_sum_qfloss.item()):<10} {_r(all_sum_actorloss.item()):<10} {_r(all_sum_adapter_loss.item()):<10} {_r(all_sum_adapter_ctx_loss.item()):<10}', )
            all_sum_qfloss = 0
            all_sum_actorloss = 0
            all_sum_alpha_loss = 0
            all_sum_adapter_loss = 0
            all_sum_adapter_ctx_loss = 0
            # Get the losses for each
            for task_index in range(len(all_my_envs)):
                data = all_my_replay_buffers[task_index].sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations, task=task_index)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions, task=task_index)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions, task=task_index)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                        min_qf_next_target
                    ).view(-1)

                qf1_a_values = qf1(data.observations, data.actions, task=task_index).view(-1)
                qf2_a_values = qf2(data.observations, data.actions, task=task_index).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss
                
                all_sum_qfloss = all_sum_qfloss + qf_loss # added here


                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations, task=task_index)
                        qf1_pi = qf1(data.observations, pi, task=task_index)
                        qf2_pi = qf2(data.observations, pi, task=task_index)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                        all_sum_actorloss = all_sum_actorloss + actor_loss
                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations, task=task_index)
                            alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                            all_sum_alpha_loss = all_sum_alpha_loss + alpha_loss

                # Here we get adapter losses and targets
                with torch.no_grad():
                    target = torch.nn.utils.parameters_to_vector(actor.heads[task_index].parameters())
                    _sp = data.next_observations
                    _s = data.observations
                    _a = data.actions
                    _r = data.rewards
                    combined_vals = torch.concat([_s, _a, _r, _sp], axis=-1)
                preds = adapter(combined_vals)
                new_target_expand = target[None].expand(len(preds), -1)
                adapter_loss_current = adapter_loss_func(preds, new_target_expand)

                all_sum_adapter_loss = all_sum_adapter_loss + adapter_loss_current
                
                preds_ctx = context_conditioned_adapter(TORCH_CTX[task_index])
                loss = adapter_loss_func(preds_ctx, target)
                all_sum_adapter_ctx_loss = all_sum_adapter_ctx_loss + loss
                    
            actor_optimizer.zero_grad()
            all_sum_actorloss.backward()
            actor_optimizer.step()
            if should_train_critic:
                q_optimizer.zero_grad()
                all_sum_qfloss.backward()
                q_optimizer.step()
            if args.autotune:
                a_optimizer.zero_grad()
                all_sum_alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()
            adapter_optimizer.zero_grad()
            all_sum_adapter_loss.backward()
            adapter_optimizer.step()
            
            adapter_optimizer_ctx.zero_grad()
            all_sum_adapter_ctx_loss.backward()
            adapter_optimizer_ctx.step()
            
            tmp_train.append({'critic_loss': all_sum_qfloss.item(), 'alpha_loss': all_sum_alpha_loss.item(), 'actor_loss': all_sum_actorloss.item(), 'adapter_loss': all_sum_adapter_loss.item()})
            
            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        if global_step >= args.total_timesteps:
            save_model(global_step, override_test='after_train_final')
        
        ALL_LOGS.append(tmp_train)
    for e in all_my_envs:
        e.close()
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
            'adapter': adapter,
            'ctx_adapter': context_conditioned_adapter
        },
        'device': device,
        'logs': ALL_LOGS
    }
