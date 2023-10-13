from collections import defaultdict
import copy
import glob
from typing import Any, Dict, List

from matplotlib import pyplot as plt
import natsort
import numpy as np
from genrlise.rlanalyse.analyse.analysis_utils import mysubplots_directly
from common.utils import mysavefig_v2, path, plot_mean_std
from genrlise.common.vars import RESULTS_DIR
from genrlise.analyse.utils import get_all_directory_seeds, get_latest_parent_from_genrl_file
from genrlise.common.infra.genrl_config import GenRLExperimentConfig

from genrlise.analyse.common import extract_specific_context_dimensions_from_result, get_context_dimension_that_changes
from genrlise.rlanalyse.common.common import get_yaml_filenames, read_in_all_results
from genrlise.rlanalyse.common.result import MeanStandardResult, Result
from genrlise.rlanalyse.plots.standard_plots import get_avg_reward,  plot_evaluation_results, plot_train_results
import seaborn as sns
from matplotlib import patches

PRETTY=True
PRETTY_LINEWIDTH = 6
PRETTY_KWARGS = dict(linewidth=3.5, alpha=0.1)
XLABEL_TRAIN = "Training Time (Steps)"
YLABEL_EVAL = "Mean Eval. Reward"


def get_load_model_kwargs(v: str) -> Dict[str, bool]:
    return dict()

def make_legend_good(leg, skiplast=False, skipn=0):
    if PRETTY:
        ll = leg.legendHandles
        if skiplast:
            ll = leg.legendHandles[:-1]
        if skipn > 0:
            ll = leg.legendHandles[:-skipn]
        for legobj in ll: legobj.set_linewidth(PRETTY_LINEWIDTH)

def plot_eval_results_from_list_of_experiments(list_of_runs_to_plot: List[str], labels: List[str], save_directory: str, 
                                               max_seeds: int = 1_000, env: str = 'ODE', plot_singles: bool = False,
                                               plot_kwargs={'have_legend': True}, do_mean_standard: bool=True,
                                               plot_each_seed_alone: bool = False,
                                               read_in_kwargs: Dict[str, Any] = {},
                                               optimal_short_name: str=None,
                                               eval_keys: List[str] = None,
                                               plot_subplots: bool = True,
                                               plot_subplots_force_size=None, read_kwargs={},
                                               override_state_1=False,
                                               title_avg=False,
                                               eval_figsize=None, do_pretty_plots=False):
    """This plots the evaluation results obtained from these runs in one plot and saves the result in {RESULTS_DIR}/{save_directory}/eval_rewards/eval_results.jpg

    Args:
        list_of_runs_to_plot (List[str]): List of strings that are short names of specific runs, e.g. list_of_runs_to_plot = ['v0145-o', 'v0180-a']
        labels (List[str]): A list, of length len(list_of_runs_to_plot), describing a human readable name for each of these runs, e.g. ['Optimal', 'SupervisedResidual']
        save_directory (str): Where this should be saved, e.g. progress/date/...
        max_seeds (int): The maximum number of seeds to consider for the plots
        env (str, optional): The environment. Either ODE or CartPole. Defaults to 'ODE'.
        plot_singles (bool, optional): If true, plots single values as well. Defaults to 'ODE'.
        plot_kwargs (dict): The arguments to send to `plot_evaluation_results`
        
        do_mean_standard (bool, optional): If true, plots the mean and standard deviation, otherwise just each seed on its own
        plot_each_seed_alone (bool, optional): If true, plots, in addition to plotting the single runs, each seed on its own plot.
        
        eval_keys (List[str], optional): The list of evaluation keys to plot. If None, then we plot all of them
    """

    
    defaults = {'optimal_short_name': None, 'do_add_baseline_results': False}
    if env == 'ODE' and optimal_short_name != -1:
        defaults['optimal_short_name'] = optimal_short_name
        if optimal_short_name is None: 
            defaults['optimal_short_name'] = 'v0201-z'
        defaults['do_add_baseline_results'] = True
        if labels is not None: 
            labels = copy.copy(labels) + ['Baseline']
    filenames = get_yaml_filenames([], list_of_runs_to_plot)
    old_results = read_in_all_results(filenames, max_seeds=max_seeds, do_mean_standard=False, select_state_1=env == "ODE" or override_state_1, read_in_kwargs=read_in_kwargs, 
                                      eval_keys=eval_keys, **defaults, **read_kwargs)

    if do_mean_standard:
        all_results = [MeanStandardResult.from_result(r) for r in old_results]
    else:
        all_results = old_results
        
    if not env.endswith("ODE"):
        all_results = [extract_specific_context_dimensions_from_result(r) for r in all_results]
    
    if 'plt_legend_ncol' not in plot_kwargs: plot_kwargs['plt_legend_ncol'] = 3
    if eval_figsize is not None:
        plt.figure(figsize=eval_figsize)
    if do_pretty_plots:
        ylims = plot_evaluation_results(all_results, folder=path(RESULTS_DIR, f'{save_directory}', 'eval_rewards'), save_name='eval_results.png', plot_ode_baseline=False, labels=labels, plt_title=None, labels_add_run_name=False, pretty=True, eval_keys=eval_keys, PRETTY_KWARGS=PRETTY_KWARGS, make_legend_good=make_legend_good, **plot_kwargs)
    else:
        ylims = plot_evaluation_results(all_results, folder=path(RESULTS_DIR, f'{save_directory}', 'eval_rewards'), save_name='eval_results.png', plot_ode_baseline=False, labels=labels, plt_title=f'Evaluation Results({env})', labels_add_run_name=False, eval_keys=eval_keys, **plot_kwargs)

    if plot_subplots:
        is_ode = (env == 'ODE') * 1
        fig, axs = mysubplots_directly(len(all_results) - 2 * is_ode, sharex=True, sharey=True, force_size=plot_subplots_force_size, additional_y_size=5)
        do_tile_rows = False
        do_tile = False
        if plot_subplots_force_size is not None and plot_subplots_force_size:
            n_res = len(all_results) - 2 * is_ode
            if plot_subplots_force_size[0] * plot_subplots_force_size[1] > n_res:
                if (plot_subplots_force_size[0] - 1) * plot_subplots_force_size[1] == n_res:
                    do_tile = False
                    do_tile_rows = True
                else:
                    do_tile = True
        else:
            do_tile = False
        
        counter = 0
        counter_2 = 0
        
        curr_t = []
        curr_labs = []
        
        if do_tile or do_tile_rows: 
            good_results_rows = [[] for _ in range(plot_subplots_force_size[1])]
            good_labels_rows = [[] for _ in range(plot_subplots_force_size[1])]
        
        for idx, (r, v, l) in enumerate(zip(all_results[is_ode:], list_of_runs_to_plot[is_ode:], labels[is_ode:])):
            counter += 1
            t = [all_results[0], r, all_results[-1]] if env == 'ODE' else [r]
            labs = [labels[0], l, labels[-1]] if env == 'ODE' else [l]
            curr_t.append(r)
            curr_labs.append(l)
            
            ax = axs[idx + counter_2]
            _test_ans = plot_evaluation_results(t, plot_ode_baseline=False, labels=labs, plt_title=f'Evaluation Results({env})', labels_add_run_name=False, set_ylim=ylims, eval_keys=eval_keys, have_legend=False, ax=ax, title_avg=title_avg, **plot_kwargs)
            if do_tile or do_tile_rows:
                col = (idx + counter_2) % plot_subplots_force_size[1]
                good_results_rows[col].append(r)
                good_labels_rows[col].append(l)
            
            ax.set_title(l)
            if title_avg:
                ax.set_title(f"{l} -- {_test_ans[-1]}")
                
            
            if do_tile and counter == plot_subplots_force_size[1] - 1:
                # Plot everything in this row on one plot
                counter = 0
                
                ax = axs[idx + counter_2 + 1]
                curr_labs = [l.split("[")[-1].replace(']', '') for l in curr_labs]
                plot_evaluation_results(curr_t, plot_ode_baseline=False, labels=curr_labs, plt_title=f'Evaluation Results({env})', labels_add_run_name=False, set_ylim=ylims, eval_keys=eval_keys, have_legend=True, ax=ax, legend_outside=True, **(plot_kwargs | {'plt_legend_ncol': 1}))
                ax.set_title("ALL")
                counter_2 += 1
                
                curr_t = []
                curr_labs = []
        if idx + counter_2 < axs.size - 1 and do_tile or do_tile_rows:
            # Plot bottom row.
            axs = axs.reshape(plot_subplots_force_size)
            for col in range(plot_subplots_force_size[1]):
                rs = good_results_rows[col]
                ls = good_labels_rows[col]
                if env == 'ODE':
                    rs = [all_results[0]] + rs + [all_results[-1]]
                    ls = [labels[0]] + ls + [labels[-1]]
                legend_outside = False
                tt = plot_kwargs
                if col == plot_subplots_force_size[1] - 1:
                    rs = all_results
                    ls = labels
                    legend_outside = True
                    tt = (plot_kwargs | {'plt_legend_ncol': 1})
                ls = [l.split('[')[0].replace(']', '') for l in ls]
                plot_evaluation_results(rs, plot_ode_baseline=False, labels=ls, plt_title=f'', labels_add_run_name=False, set_ylim=ylims, eval_keys=eval_keys, have_legend=True, ax=axs[-1, col], legend_outside=legend_outside, **tt)
                
        plt.tight_layout()
        mysavefig_v2((path(RESULTS_DIR, f'{save_directory}', 'eval_rewards'), f'all_on_one.png'));
        plt.close()
        
    if plot_singles:
        for idx, (r, v, l) in enumerate(zip(all_results[:], list_of_runs_to_plot[:], labels[:])):
            t = [all_results[0], r, all_results[-1]] if env == 'ODE' else [r]
            labs = [labels[0], l, labels[-1]] if env == 'ODE' else [l]
            plot_evaluation_results(t, folder=path(RESULTS_DIR, f'{save_directory}', 'eval_rewards', 'single'), save_name=f'{v}.png', plot_ode_baseline=False, labels=labs, plt_title=f'Evaluation Results({env})', labels_add_run_name=False, set_ylim=ylims, eval_keys=eval_keys, **plot_kwargs)
                       
            if plot_each_seed_alone:
                for key in all_results[0].evaluation_metrics.keys():
                    r = old_results[idx]
                    num_seeds = len(r.evaluation_metrics[key].contexts)
                    fig, axs = mysubplots_directly(num_seeds)
                    if num_seeds == 1:
                        axs = np.array([axs])
                    for seed, ax in zip(range(num_seeds), axs.ravel()):
                        t = [old_results[0], r, old_results[-1]] if env == 'ODE' else [r]
                        labs = [labels[0], l, labels[-1]] if env == 'ODE' else [l]
                        
                        t = [_r.select_seed(seed) for _r in t]
                        plot_evaluation_results(t, plot_ode_baseline=False, labels=labs, plt_title=f'Evaluation Results({env}) Seed ({seed})', labels_add_run_name=False, set_ylim=ylims, ax=ax, add_seed_to_label=False, have_legend=seed == num_seeds-1, **plot_kwargs)
                    
                    save_name=f'{v}_seeds'
                    folder=path(RESULTS_DIR, f'{save_directory}', 'eval_rewards', 'single_seeds', f'{save_name}')
                    mysavefig_v2((folder, f'{key}.png'));
                    plt.close()
    

def plot_eval_summary_distance_to_training_context(list_of_runs_to_plot: List[str], labels: List[str], save_directory: str, 
                                               max_seeds: int = 1_000, env: str = 'ODE', plot_singles: bool = False,
                                               plot_kwargs={'have_legend': True}, do_mean_standard: bool=True,
                                               plot_each_seed_alone: bool = False,
                                               read_in_kwargs: Dict[str, Any] = {},
                                               optimal_short_name: str=None,
                                               eval_keys: List[str] = None,
                                               plot_subplots: bool = True,
                                               plot_subplots_force_size=None, read_kwargs={},
                                               override_state_1=False,
                                               do_scatter=False):

    defaults = {'optimal_short_name': None, 'do_add_baseline_results': False}
    if env == 'ODE' and optimal_short_name != -1:
        defaults['optimal_short_name'] = optimal_short_name
        if optimal_short_name is None: 
            defaults['optimal_short_name'] = 'v0201-z'
        defaults['do_add_baseline_results'] = True
        if labels is not None: 
            labels = copy.copy(labels) + ['Baseline']
        
    filenames = get_yaml_filenames([], list_of_runs_to_plot)
    old_results = read_in_all_results(filenames, max_seeds=max_seeds, do_mean_standard=False, select_state_1=env == "ODE" or override_state_1, read_in_kwargs=read_in_kwargs, 
                                      eval_keys=eval_keys, include_train_data=True, **defaults, **read_kwargs)
    
    all_results = old_results
    
    if 'plt_legend_ncol' not in plot_kwargs: plot_kwargs['plt_legend_ncol'] = 3
    if plot_subplots:
        is_ode = (env == 'ODE') * 1
        fig, axs = mysubplots_directly(len(all_results) - 2 * is_ode, sharex=True, sharey=True, force_size=plot_subplots_force_size, additional_y_size=5)
        do_tile_rows = False
        do_tile = False
        if plot_subplots_force_size is not None and plot_subplots_force_size:
            n_res = len(all_results) - 2 * is_ode
            if plot_subplots_force_size[0] * plot_subplots_force_size[1] > n_res:
                if (plot_subplots_force_size[0] - 1) * plot_subplots_force_size[1] == n_res:
                    do_tile = False
                    do_tile_rows = True
                else:
                    do_tile = True
        else:
            do_tile = False
        
        counter = 0
        counter_2 = 0
        
        curr_t = []
        curr_labs = []
        
        if do_tile or do_tile_rows: 
            good_results_rows = [[] for _ in range(plot_subplots_force_size[1])]
            good_labels_rows = [[] for _ in range(plot_subplots_force_size[1])]
            good_x_rows = [[] for _ in range(plot_subplots_force_size[1])]
            good_std_rows = [[] for _ in range(plot_subplots_force_size[1])]
        
        all_exps = []
        for idx, (r, v, l) in enumerate(zip(all_results[is_ode:], list_of_runs_to_plot[is_ode:], labels[is_ode:])):
            counter += 1
            t = [all_results[0], r, all_results[-1]] if env == 'ODE' else [r]
            labs = [labels[0], l, labels[-1]] if env == 'ODE' else [l]
            curr_labs.append(l)
            assert len(eval_keys) == 1
            key = eval_keys[0]
            ax = axs[idx + counter_2]
            train_ctxs = np.unique(r.train_metrics.contexts[0], axis=0)  # Shape is (N, C)
            eval_cs = r.evaluation_metrics[key].contexts # shape is something like (seeds, episodes, C)
            ttt = train_ctxs[:, None, None] - eval_cs
            norms = np.linalg.norm(ttt, axis=-1)
            min_dist = np.min(norms, axis=0)
            
            rewards = r.evaluation_metrics[key].rewards
            x_val = min_dist.mean(axis=0)
            y_val = rewards.mean(axis=0)
            y_std = rewards.std(axis=0)
            def rem(arr, mins, maxs):
                return arr
            A, B, C = rem(x_val, 90, 112), rem(y_val, 90, 112), rem(y_std, 90, 112)
            
            uniq = np.unique(A)
            tmp_new_dist = []
            tmp_new_rews = []
            tmp_new_rews_std = []
            for u in uniq:
                idxs = np.isclose(u[None], A)#.all(axis=-1)
                tmp_new_dist.append(t:=A[idxs].reshape((idxs.sum())).mean(axis=0))
                tmp_new_rews.append(B[idxs].reshape((idxs.sum())).mean(axis=0))
                tmp_new_rews_std.append(C[idxs].reshape((idxs.sum())).mean(axis=0))
            all_exps.append((tmp_new_dist, tmp_new_rews, tmp_new_rews_std, l))
            if do_scatter:
                ax.scatter(tmp_new_dist, tmp_new_rews)
            else:
                plot_mean_std(tmp_new_rews, tmp_new_rews_std, x=tmp_new_dist, ax=ax)
            curr_t.append((tmp_new_dist, tmp_new_rews, tmp_new_rews_std))
            ax.set_xlabel("Distance to Closest Training Context")
            ax.set_ylabel("Reward")

            if do_tile or do_tile_rows:
                col = (idx + counter_2) % plot_subplots_force_size[1]
                good_results_rows[col].append(tmp_new_rews)
                good_x_rows[col].append(tmp_new_dist)
                good_labels_rows[col].append(l)
                good_std_rows[col].append(tmp_new_rews_std)
            
            _a = np.array
            ax.set_title(f"{l} -- D*R = {np.round(np.mean(_a(tmp_new_dist) * _a(tmp_new_rews)))} -- R = {np.round(np.mean(_a(tmp_new_rews)))}")
            
            if do_tile and counter == plot_subplots_force_size[1] - 1:
                # Plot everything in this row on one plot
                counter = 0
                
                ax = axs[idx + counter_2 + 1]
                curr_labs = [l.split("[")[-1].replace(']', '') for l in curr_labs]
                for t, l in zip(curr_t, curr_labs):
                    if do_scatter:
                        ax.scatter(t[0], t[1], label=l)
                    else:
                        plot_mean_std(t[1], t[2], x=t[0], ax=ax, label=l)
                ax.legend()
                ax.set_title("ALL")
                ax.set_xlabel("Distance to Closest Training Context")
                ax.set_ylabel("Reward")
                counter_2 += 1
                
                curr_t = []
                curr_labs = []
        if idx + counter_2 < axs.size - 1 and do_tile or do_tile_rows:
            # Plot bottom row.
            axs = axs.reshape(plot_subplots_force_size)
            for col in range(plot_subplots_force_size[1] - 1):
                rs = good_results_rows[col]
                xs = good_x_rows[col]
                stds = good_std_rows[col]
                ls = good_labels_rows[col]
                legend_outside = False
                tt = plot_kwargs
                if col == plot_subplots_force_size[1] - 1:
                    rs = all_results
                    ls = labels
                    legend_outside = True
                    tt = (plot_kwargs | {'plt_legend_ncol': 1})
                ls = [l.split('[')[0].replace(']', '') for l in ls]
                
                for r, x, std, l in zip(rs, xs, stds, ls):
                    if do_scatter:
                        axs[-1, col].scatter(x, r, label=l)
                        axs[-1, -1].scatter(x, r, label=l)
                    else:
                        plot_mean_std(r, std, x=x, ax=axs[-1, col], label=l)
                        plot_mean_std(r, std, x=x, ax=axs[-1, -1], label=l)
                axs[-1, col].legend()
            axs[-1, -1].legend()

                

        name = ''
        if do_scatter:
            name = '_scatter'
        plt.tight_layout()
        mysavefig_v2((path(RESULTS_DIR, f'{save_directory}', 'summary_results'), f'all_on_one{name}.png'));
        plt.close()
        
        for tmp_new_dist, tmp_new_rews, tmp_new_rews_std, l in all_exps:
            if do_scatter:
                plt.scatter(tmp_new_dist, tmp_new_rews, label=l)
            else:
                plot_mean_std(tmp_new_rews, tmp_new_rews_std, x=tmp_new_dist, label=l)
        plt.xlabel("Distance to Closest Training Context")
        plt.ylabel("Reward")
        plt.legend()
        
        
        plt.tight_layout()

        mysavefig_v2((path(RESULTS_DIR, f'{save_directory}', 'summary_results'), f'single_plot{name}.png'));
        plt.close()
        

def plot_eval_summary_auc_vs_train(            list_of_runs_to_plot: List[str], labels: List[str], save_directory: str, 
                                               model_names: List[str], train_times: List[int],
                                               max_seeds: int = 1_000, env: str = 'ODE', plot_singles: bool = False,
                                               plot_kwargs={'have_legend': True}, do_mean_standard: bool=True,
                                               plot_each_seed_alone: bool = False,
                                               read_in_kwargs: Dict[str, Any] = {},
                                               optimal_short_name: str=None,
                                               eval_keys: List[str] = None,
                                               plot_subplots: bool = True,
                                               plot_subplots_force_size=None, read_kwargs={},
                                               override_state_1=False,
                                               do_checkpoints=False, eval_figsize=None,
                                               restrict_context=None,
                                               specific=None,
                                               train_ctxs = [-5, -1, 1, 5],
                                               have_legend = True,
                                               legend_outside=True,
                                               legend_under=False,
                                               legend_title=None,
                                               legend_ncol=5,
                                               max_checkpoint=None,
                                               colours=None,
                                               ignore_specific_dimensions=False,
                                               return_if_multiple=False,
                                               do_ylim=None,
                                               other_ylabel=YLABEL_EVAL,
                                               is_specific_multiple_dims=False,
                                               return_all=False
                                               ):

    defaults = {'optimal_short_name': None, 'do_add_baseline_results': False}
    if env == 'ODE' and optimal_short_name != -1:
        defaults['optimal_short_name'] = optimal_short_name
        if optimal_short_name is None: 
            defaults['optimal_short_name'] = 'v0201-z'
        defaults['do_add_baseline_results'] = True
        if labels is not None: 
            labels = copy.copy(labels) + ['Baseline']
    
    
    if do_checkpoints:
        train_times = []
        good_results   = []
        good_labels = []
        filenames = get_yaml_filenames([], list_of_runs_to_plot)
        for val, file, label in zip(list_of_runs_to_plot, filenames, labels):
            parent = get_latest_parent_from_genrl_file(file)
            exp_conf = GenRLExperimentConfig(file)
            filename = get_all_directory_seeds(parent)[0]
            tt = int(exp_conf("train/steps"))
            for n in natsort.natsorted(glob.glob(f"{filename}/checkpoints/*")):
                checkpoint = n.split("/")[-1]
                if '_test' in checkpoint or (int(checkpoint) >= tt and tt != 0): 
                    continue
                if max_checkpoint is not None and int(checkpoint) > max_checkpoint: break
                if val == 'v0801a-fb5' and max_checkpoint == 300_000 and int(checkpoint) == 300_000: break
                good_results.append(Result.load_in(file, max_seeds=max_seeds, include_train_data=False, read_in_kwargs=read_in_kwargs, start_seeds=0, checkpoint=checkpoint, eval_keys=eval_keys))
                good_labels.append(label)
                train_times.append(int(checkpoint))
            
            
            is_bad = max_checkpoint is not None and len(train_times) > 0 and train_times[-1] >= max_checkpoint
            if not is_bad and (max_checkpoint is None or tt < max_checkpoint):
                good_labels.append(label)
                if tt == 0 and len(train_times) >= 2:
                    # Make this correct
                    dd = train_times[-1] - train_times[-2]
                    train_times.append(dd + train_times[-1])
                else:
                    train_times.append(tt)
                good_results.append(Result.load_in(file, max_seeds=max_seeds, include_train_data=False, read_in_kwargs=read_in_kwargs, start_seeds=0, eval_keys=eval_keys))
        old_results = good_results
        model_names = good_labels
    else:
        filenames = get_yaml_filenames([], list_of_runs_to_plot)
        old_results = read_in_all_results(filenames, max_seeds=max_seeds, do_mean_standard=False, select_state_1=env == "ODE" or override_state_1, read_in_kwargs=read_in_kwargs, 
                                      eval_keys=eval_keys, include_train_data=True, **defaults, **read_kwargs)
    
    all_results = old_results
        
    if env in ["CartPole", 'ODE2', 'XY_CartPole', 'BoundedODE2']:
        if not ignore_specific_dimensions:
            all_results = [extract_specific_context_dimensions_from_result(r, return_if_multiple=return_if_multiple) for r in all_results]
    
    if 'plt_legend_ncol' not in plot_kwargs: plot_kwargs['plt_legend_ncol'] = 3
    all_d = defaultdict(lambda: [])
    all_x = defaultdict(lambda: [])
    all_y = defaultdict(lambda: [])
    all_std = defaultdict(lambda: [])
    all_everything = defaultdict(lambda: [])
    assert len(eval_keys) == 1
    key = eval_keys[0]
    for r, name, train in zip(all_results, model_names, train_times):
        all_d[name].append(r)
        all_x[name].append(train)
        # Get the y value
        rewards = r.evaluation_metrics[key].rewards
        ctxs = r.evaluation_metrics[key].contexts
        if restrict_context is not None:
            if len(ctxs.shape) == 3:
                idx_to_use = np.logical_and(ctxs[0, :, 0] >= restrict_context[0], ctxs[0, :, 0] <= restrict_context[1])
            else:
                idx_to_use = np.logical_and(ctxs[0, :] >= restrict_context[0], ctxs[0, :] <= restrict_context[1])
            ctxs = ctxs[:, idx_to_use]
            rewards = rewards[:, idx_to_use]
        minmax = None
        if specific is not None:
            _eps = 0.05
            train_ctxs = np.array(train_ctxs)
            if is_specific_multiple_dims:
                A = np.sign(np.round(ctxs[:, :, None], 4) - train_ctxs[None, None, :])
            else:
                ctxs = ctxs.squeeze(-1)
                # round to get numerical things in check
                A = np.sign(np.round(ctxs[:, :, None], 4) - train_ctxs[None, None, :])
            minmax = (0, 210)
            og_r = rewards
            og_c = ctxs
            if specific == 'train':
                if is_specific_multiple_dims:
                    new_idxs = np.min(np.linalg.norm(np.abs(ctxs[:, :, None] - train_ctxs[None, None, :]), axis=-1) , axis=-1) < _eps
                else:
                    new_idxs = np.min(np.abs(ctxs[:, :, None] - train_ctxs[None, None, :]) , axis=-1) < _eps
            elif specific == 'interpolation':
                if is_specific_multiple_dims:
                    new_idxs = np.logical_and((A < 0).any(axis=-2), (A > 0).any(axis=-2)).all(axis=-1)
                else:
                    new_idxs = np.logical_and((A < 0).any(axis=-1), (A > 0).any(axis=-1))
            elif specific == 'extrapolation':
                if is_specific_multiple_dims:
                    idx_my_a = (A < 0).all(axis=-2).all(axis=-1)
                    idx_my_b = (A > 0).all(axis=-2).all(axis=-1)
                else:
                    idx_my_a = (A < 0).all(axis=-1)
                    idx_my_b = (A > 0).all(axis=-1)
                new_idxs = np.logical_or(idx_my_a, idx_my_b)
            
            def do_reshape(idx, c, r):
                if is_specific_multiple_dims:
                    r    = r[idx].reshape(len(c), -1)
                    c    = c[idx].reshape(len(c), -1, c.shape[-1])
                else:
                    r    = r[idx].reshape(len(c), -1)
                    c    = c[idx].reshape(len(c), -1)
                return r, c
            
            rewards, ctxs = do_reshape(new_idxs, ctxs, rewards)

        if rewards.shape[-1] == 1 and len(rewards.shape) >= 3:
            rewards = rewards.squeeze(-1)
        # This calculates the average reward over time.

        vals = get_avg_reward(rewards, ctxs)
        if specific == 'extrapolation':
            v1 = get_avg_reward(*do_reshape(idx_my_a, og_c, og_r))
            v2 = get_avg_reward(*do_reshape(idx_my_b, og_c, og_r))
            newvals = (v1 + v2) / 2
            vals = newvals
            
        assert len(vals.shape) == 1
        all_y[name].append(np.mean(vals))
        all_std[name].append(np.std(vals))
        all_everything[name].append(vals)
    fig, axs = mysubplots_directly(len(all_d), sharex=True, sharey=True, force_size=(1, len(all_d) + 1))
    for index, (name, ax, label) in enumerate(zip(all_d.keys(), axs.ravel(), labels)):
        ys = []
        c = {} if colours is None else {'color': colours[index]}
        plot_mean_std(all_y[name], all_std[name], x=all_x[name], label=label, ax=ax, **c, **PRETTY_KWARGS)
        ax.legend()
        
        plot_mean_std(all_y[name], all_std[name], x=all_x[name], label=label, ax=axs[-1], **c, **PRETTY_KWARGS)
    axs[-1].legend()
    for a in axs: a.set_xlabel(XLABEL_TRAIN)
    axs[0].set_ylabel(other_ylabel)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if do_ylim is not None:
        axs[0].set_ylim(do_ylim[0], do_ylim[1])
    plt.tight_layout()
    my_extra = '' if specific is None else f'_{specific}'
    mysavefig_v2((path(RESULTS_DIR, f'{save_directory}', 'summary_results'), f'vs_train_time{my_extra}.png'));
    plt.close()
    
    if eval_figsize is not None:
        plt.figure(figsize=eval_figsize)
    for index, (name, label) in enumerate(zip(all_d.keys(), labels)):
        c = {} if colours is None else {'color': colours[index]}
        plot_mean_std(all_y[name], all_std[name], x=all_x[name], label=label, **c, **PRETTY_KWARGS)
    if have_legend:
        if legend_under:
            has_title = legend_title is not None
            
            if eval_figsize == (10, 4):
                leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.12 + has_title * 0.05), fancybox=True, shadow=True, ncol=legend_ncol, title=legend_title)
            else:
                leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1 + has_title * 0.05), fancybox=True, shadow=True, ncol=legend_ncol, title=legend_title)
        elif legend_outside:
            leg = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', title=legend_title)
        else:
            leg = plt.legend()
        make_legend_good(leg)
    if minmax is not None:
        plt.ylim(minmax[0], minmax[1])
    ytext = YLABEL_EVAL
    if specific is not None: ytext = YLABEL_EVAL
    plt.ylabel(ytext)
    plt.xlabel(XLABEL_TRAIN)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    mysavefig_v2((path(RESULTS_DIR, f'{save_directory}', 'summary_results'), f'vs_train_time_single_plot{my_extra}.png'));
    plt.close()
    if return_all:
        return all_x, all_y, all_std, labels, all_everything
    return all_x, all_y, all_std, labels

def plot_train_results_from_list_of_experiments(list_of_runs_to_plot: List[str], labels: List[str], save_directory: str, 
                                               max_seeds: int = 1_000, env: str = 'ODE', plot_singles: bool = False,
                                               plot_kwargs={}, do_mean_standard: bool=True,
                                               plot_each_seed_alone: bool = False,
                                               optimal_short_name: str=None,
                                               eval_keys: List[str] =None,
                                               plot_subplots=False,
                                               plot_subplots_force_size=None,
                                               read_kwargs={}):

    
    defaults = {'optimal_short_name': None, 'do_add_baseline_results': False}

    filenames = get_yaml_filenames([], list_of_runs_to_plot)
    old_results = read_in_all_results(filenames, max_seeds=max_seeds, do_mean_standard=False, select_state_1=False, include_train_data=True, eval_keys=eval_keys, **defaults, **read_kwargs)
    
    if do_mean_standard:
        all_results = [MeanStandardResult.from_result(r) for r in old_results]
    else:
        all_results = old_results
        
    if env == "CartPole":
        all_results = [extract_specific_context_dimensions_from_result(r) for r in all_results]
    
    if 'plt_legend_ncol' not in plot_kwargs: plot_kwargs['plt_legend_ncol'] = 3
    ylims = plot_train_results(all_results, folder=path(RESULTS_DIR, f'{save_directory}', 'train_rewards'), save_name='train_results.png', plot_ode_baseline=False, labels=labels, plt_title=f'Train Results({env})', labels_add_run_name=False, **plot_kwargs)
    plot_kwargs['set_ylim'] = ylims
    if plot_subplots:
        is_ode = (env == 'ODE') * 1
        do_tile_rows = False
        do_tile = False
        if plot_subplots_force_size is not None and plot_subplots_force_size:
            n_res = len(all_results) - 2 * is_ode
            if plot_subplots_force_size[0] * plot_subplots_force_size[1] > n_res:
                if (plot_subplots_force_size[0] - 1) * plot_subplots_force_size[1] == n_res:
                    do_tile = False
                    do_tile_rows = True
                else:
                    do_tile = True
        else:
            do_tile = False
        
        counter = 0
        counter_2 = 0
        
        curr_t = []
        curr_labs = []
        
        if do_tile or do_tile_rows: 
            good_results_rows = [[] for _ in range(plot_subplots_force_size[1])]
            good_labels_rows = [[] for _ in range(plot_subplots_force_size[1])]
        
        fig, axs = mysubplots_directly(len(all_results) - 2 * is_ode, sharex=True, sharey=True, force_size=plot_subplots_force_size)
        for idx, (r, v, l) in enumerate(zip(all_results[is_ode:], list_of_runs_to_plot[is_ode:], labels[is_ode:])):
            counter += 1
            t = [all_results[0], r] if is_ode else [r]
            labs = [labels[0], l] if is_ode else [l]
            curr_t.append(r)
            curr_labs.append(l)
            
            ax = axs[idx + counter_2]
            plot_train_results(t, plot_ode_baseline=False, labels=labs, plt_title=f'Train Results({env})', labels_add_run_name=False, have_legend=False, ax=ax, **plot_kwargs)
            if do_tile or do_tile_rows:
                col = (idx + counter_2) % plot_subplots_force_size[1]
                good_results_rows[col].append(r)
                good_labels_rows[col].append(l)
            
            ax.set_title(l)
                        
            if do_tile and counter == plot_subplots_force_size[1] - 1:
                # Plot everything in this row on one plot
                counter = 0
                
                ax = axs[idx + counter_2 + 1]
                curr_labs = [l.split("[")[-1].replace(']', '') for l in curr_labs]
                plot_train_results(curr_t, plot_ode_baseline=False, labels=curr_labs, plt_title=f'Train Results({env})', labels_add_run_name=False, have_legend=True, legend_outside=True, ax=ax, **(plot_kwargs | {'plt_legend_ncol': 1}))
                ax.set_title("ALL")
                counter_2 += 1
                
                curr_t = []
                curr_labs = []
            
            

        if idx + counter_2 < axs.size - 1 and do_tile or do_tile_rows:
            # Plot bottom row.
            axs = axs.reshape(plot_subplots_force_size)
            for col in range(plot_subplots_force_size[1]):
                rs = good_results_rows[col]
                ls = good_labels_rows[col]
                if env == 'ODE':
                    rs = [all_results[0]] + rs + [all_results[-1]]
                    ls = [labels[0]] + ls + [labels[-1]]
                legend_outside = False
                tt = plot_kwargs
                if col == plot_subplots_force_size[1] - 1:
                    rs = all_results
                    ls = labels
                    legend_outside = True
                    tt = (plot_kwargs | {'plt_legend_ncol': 1})
                ls = [l.split('[')[0].replace(']', '') for l in ls]
                plot_train_results(rs, plot_ode_baseline=False, labels=ls, plt_title=f'Train Results({env})', labels_add_run_name=False, have_legend=True, ax=axs[-1, col], legend_outside=legend_outside, **tt)
                
        plt.suptitle(f"Training Rewards ({env})")
        mysavefig_v2((path(RESULTS_DIR, f'{save_directory}', 'train_rewards'), f'all_on_one.png'));
        plt.close()

def plot_cartesian_product(vals, 
                           folder,
                           key: str = 'TestCartesianProduct', 
                           min_rew=0, max_rew=500,
                           first_label = 'Cart Mass',
                           second_label = 'Pole Length',
                           read_kwargs={},
                           do_convex_hull_marking=True, cmap=None, do_pairwise_diffs=False,
                           get_train_from_yaml=False, div_1k=False,
                           figsize=(10,10), use_idxs_top_2=False):
    filenames = get_yaml_filenames([], vals)
    defaults = {'optimal_short_name': None, 'do_add_baseline_results': False}
    old_results = read_in_all_results(filenames, max_seeds=8, do_mean_standard=False, select_state_1=False, read_in_kwargs={}, eval_keys=[key], include_train_data=True, **defaults, **read_kwargs)
    all_results = [MeanStandardResult.from_result(r) for r in old_results]
    all_results_v06 = None
    if 'v08' in vals[0]:
        # Some patching
        tt = read_in_all_results([f.replace('v08', 'v06') for f in filenames], max_seeds=8, do_mean_standard=False, select_state_1=False, read_in_kwargs={}, eval_keys=[key], include_train_data=True, **defaults, skip_asserts=True, **read_kwargs)
        all_results_v06 = [MeanStandardResult.from_result(r) for r in tt]
        
    
    all_arrs = []
    def plot_heatmap(arr, v, is_diff=False):
        plt.figure(figsize=figsize)
        if not is_diff:
            sns.heatmap(arr, annot=True, xticklabels=xticks, yticklabels=yticks, fmt='g', cbar=False, vmin=min_rew, vmax=max_rew, cmap=cmap)
        else:
            sns.heatmap(arr, annot=True, xticklabels=xticks, yticklabels=yticks, fmt='g', cbar=False, vmin=min_rew, vmax=max_rew, cmap=cmap)
        
        def add_patch(ax, x, y, w, h, col='green'):
            ax.add_patch(
                patches.Rectangle(
                    (x, y), w, h,
                    edgecolor=col,
                    fill=False,
                    lw=2
                ))
        if do_convex_hull_marking is not None:
            ax = plt.gca()
            do_normal = do_convex_hull_marking == False
            if do_convex_hull_marking == 'both' or do_normal:
                for a in train_ctx:
                    x = np.argmin(np.abs(all_xvals - a[0]))
                    y = np.argmin(np.abs(all_yvals - a[1]))
                    add_patch(ax, x, y, 1, 1, col='blue')
            if do_convex_hull_marking == 'both' or do_convex_hull_marking:
                add_patch(ax, minmax_idx_x[0], minmax_idx_y[0],
                              minmax_idx_x[1] - minmax_idx_x[0],
                              minmax_idx_y[1] - minmax_idx_y[0])
        
        plt.xlabel(second_label)
        plt.ylabel(first_label)
        plt.title(f"Average = {np.round(arr.mean())}")
        plt.tight_layout()
        mysavefig_v2((folder, v + ".png"))
        plt.close()
    for iii, (r, v) in enumerate(zip(all_results, vals)):
        
        r_to_use = r
        if r_to_use.train_metrics.contexts[0].shape[0] == 0 and all_results_v06 is not None:
            r_to_use = all_results_v06[iii]
        
        if get_train_from_yaml:
            yy = GenRLExperimentConfig(filenames[iii])
            ll = yy("train/context/list_of_contexts")
            train_ctx = np.array(ll)
            
        if r_to_use.train_metrics.contexts[0].shape[-1] != 2:
            indexes = get_context_dimension_that_changes(r_to_use, key, return_if_multiple=True)
            if use_idxs_top_2: indexes = indexes[:2]
            assert len(indexes) == 2
            if not get_train_from_yaml: train_ctx = r_to_use.train_metrics.contexts[0][:, indexes]
        else:
            indexes = [0, 1]
            if not get_train_from_yaml: train_ctx = r_to_use.train_metrics.contexts[0]
        if get_train_from_yaml:
            train_ctx = train_ctx[:, indexes]
        if do_convex_hull_marking is not None:
            minmax_y = np.min(train_ctx[:, 0]), np.max(train_ctx[:, 0])
            minmax_x = np.min(train_ctx[:, 1]), np.max(train_ctx[:, 1])
        
        M = r.evaluation_metrics[key]
        C = M.contexts[0, :][:, indexes]
        R = M.rewards[0]
        mins, maxs = (C[0], C[-1])
        mins = C[:, 0].min(), C[:, 1].min()
        maxs = C[:, 0].max(), C[:, 1].max()
        all_xvals = np.round(np.unique(C[:, 1]), 2).tolist()
        all_yvals = np.round(np.unique(C[:, 0]), 2).tolist()
        
        if do_convex_hull_marking is not None:
            # find min max indexes in xvals and yvals.
            minmax_idx_x = np.argmin(np.abs(all_xvals - minmax_x[0])), np.argmin(np.abs(all_xvals - minmax_x[1]))+1
            minmax_idx_y = np.argmin(np.abs(all_yvals - minmax_y[0])), np.argmin(np.abs(all_yvals - minmax_y[1]))+1
        bins = (len(all_yvals), len(all_xvals))
        arr = np.zeros(bins) - 1
        for c, r in zip(C, R):
            i = np.argmin(np.abs(all_yvals - c[0]))
            j = np.argmin(np.abs(all_xvals - c[1]))
            arr[i, j] = int(r)

        
        xticks = list(map(lambda x:f"{x:.2g}", all_xvals)) + ['avg']
        yticks = list(map(lambda x:f"{x:.2g}", all_yvals)) + ['avg']
        
        test = np.empty((arr.shape[0] + 1, arr.shape[1] + 1))
        test[:-1, :-1] = arr
        test[:-1, -1]    = arr.mean(axis=1)
        test[-1, :-1]    = arr.mean(axis=0) 
        test[-1, -1]     = arr.mean() 
        
        test[-1, :] = np.round(test[-1, :])
        test[:, -1] = np.round(test[:, -1])
        arr = test
        if div_1k:
            arr = arr / 1000
        plot_heatmap(arr, v)        
        all_arrs.append(arr)

    if do_pairwise_diffs:
        for i, v1 in enumerate(vals):
            for j, v2 in enumerate(vals):
                if i == j: continue
                if (v1 == 'v0674b-fb4' and v2 == 'v0674b-bb3') or 'v0674' not in v1:
                    diff = all_arrs[i] - all_arrs[j]
                    plot_heatmap(diff, f"sub_{v1}-{v2}", is_diff=True)
