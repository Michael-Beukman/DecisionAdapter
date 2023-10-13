from typing import List

from matplotlib import pyplot as plt
import numpy as np
from genrlise.rlanalyse.analyse.analysis_utils import get_subplot_size, mysubplots
from common.utils import mysavefig_v2, plot_mean_std
from genrlise.rlanalyse.common.result import MeanStandardResult, Result
import seaborn as sns
import scipy.integrate as integrate
# ================ Several Plotting Functions ================ 
sns.set_theme()

def get_avg_reward(reward, contexts, round=False):
    # Computes the average/integral reward
    if reward.size != contexts.size:
        if contexts.shape[1:] in [(4000, 5), (8000, 5)]: # Hard code for a particular scenario, to ensure it works correctly.
            contexts = contexts[:, :, :2]
            AA =  np.mean(reward, axis=-1)
            return AA
        alls = []
        for seed in range(len(reward)):
            assert contexts.shape[-1] == 2
            C = contexts[seed]
            Ys = C[:, 1]
            Rs = reward[seed]

            updated_integral = []
            updated_y        = []
            for y in np.unique(Ys):
                idx = Ys == y
                
                tmp_x_val = C[idx, 0]
                tmp_r     = (Rs[idx])
                if len(tmp_x_val) == 1: 
                    a = tmp_r[0]
                else:
                    a = integrate.simpson(tmp_r, tmp_x_val) / (np.max(tmp_x_val) - np.min(tmp_x_val))
                updated_integral.append(a)
                updated_y.append(y)
            if len(updated_integral) == 1:
                new_val = updated_integral[0]
            else:
                new_val = integrate.simpson(updated_integral, updated_y) / (np.max(updated_y) - np.min(updated_y))
            alls.append(new_val)
        alls = np.array(alls)
        if round: alls = np.round(alls)
        return alls
        
    R = reward.reshape(reward.shape)
    C = contexts.reshape(reward.shape)
    a = integrate.simpson(R, C) / (np.max(contexts) - np.min(contexts))
    if round: a = np.round(a)
    return a

def plot_evaluation_results(list_of_results: List[MeanStandardResult], 
                            folder: str = None, save_name: str = None, 
                            plot_ode_baseline: bool = True, ax=None,
                            plot_yscale='symlog',
                            ax_vlines=[-1, 1],
                            plt_xaxis_label='Context',
                            plt_yaxis_label='Reward',
                            plt_title='Various Evaluation Results',
                            plt_legend_ncol=1,
                            labels: List[str] = None,
                            labels_add_run_name=True,
                            set_ylim=None,
                            have_legend: bool = True,
                            add_seed_to_label: bool = True,
                            eval_keys: List[str] = None,
                            legend_outside = False,
                            title_avg=False,
                            pretty=False,
                            PRETTY_KWARGS=None, make_legend_good=None):
    import seaborn as sns; sns.set_theme()
    keys_to_iterate_over = list_of_results[0].evaluation_metrics.keys() if eval_keys is None else eval_keys
    og_ax = ax
    for eval_key in keys_to_iterate_over:
        if og_ax is None: ax = plt.gca()
        if labels is not None: assert len(list_of_results) == len(labels)
        for index, result in enumerate(list_of_results):
            if title_avg: 
                assert len(list_of_results) == 1
                mean_score = get_avg_reward(result.evaluation_metrics[eval_key].rewards[0], result.evaluation_metrics[eval_key].contexts[0], round=True)
            
            label = result.yaml_config('meta/run_name')
            if labels is not None:
                label = f"{labels[index]}({label})" if labels_add_run_name else labels[index]
            if isinstance(result, Result) and not isinstance(result, MeanStandardResult):
                for i in range(len(result.evaluation_metrics[eval_key].rewards)):
                    label_to_plot = f"{label} - {i}" if add_seed_to_label else label
                    ax.plot(result.evaluation_metrics[eval_key].contexts[0], result.evaluation_metrics[eval_key].rewards[i], label=label_to_plot)
                    if label.lower() in ['optimal', 'baseline']: break
            else:
                if pretty:
                    c = PRETTY_KWARGS
                else:
                    c = {}
                plot_mean_std(result.evaluation_metrics[eval_key].rewards[0], result.evaluation_metrics[eval_key].rewards[1], x=result.evaluation_metrics[eval_key].contexts[0], label=label, ax=ax, **c)
        if plot_ode_baseline: ax.plot(result.evaluation_metrics[eval_key].contexts[0], result.evaluation_metrics[eval_key].states[0] ** 2 * -200, label='baseline')
        if have_legend:
            if pretty:
                leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=3)
                make_legend_good(leg)
            elif legend_outside:
                ax.legend(ncol=plt_legend_ncol, bbox_to_anchor=(1, 1), loc='upper left')
            else:
                ax.legend(ncol=plt_legend_ncol)
        ax.set_xlabel(plt_xaxis_label)
        ax.set_ylabel(plt_yaxis_label)
        if plt_title is not None: ax.set_title(plt_title)
        ax.set_yscale(plot_yscale)
        if set_ylim is not None: ax.set_ylim(*set_ylim)
        _a, _b = ax.get_ylim()
        if pretty:
            newa = ((_a) + (_b - _a) * 0.1)
            if 0 <= newa <= 10: newa = 0
            ax.vlines(ax_vlines, newa, _b, color='red')
        else:
            ax.vlines(ax_vlines, _a, _b, color='red')
        if save_name is not None and folder is not None:
            mysavefig_v2((folder, f'{save_name.replace(".png", "")}_{eval_key}.png'));
            plt.close()
    if title_avg: return _a, _b, mean_score
    return _a, _b
        

def plot_train_results(list_of_results: List[MeanStandardResult], 
                            folder: str = None, save_name: str = None, 
                            plot_ode_baseline: bool = True, ax=None,
                            plot_yscale='linear',
                            ax_vlines=[-1, 1],
                            plt_xaxis_label='Episodes',
                            plt_yaxis_label='Reward',
                            plt_title='Various Train Results',
                            plt_legend_ncol=1,
                            labels: List[str] = None,
                            labels_add_run_name=True,
                            set_ylim=None,
                            have_legend: bool = True,
                            add_seed_to_label: bool = True,
                            legend_outside = False):
    import seaborn as sns; sns.set_theme()

    if ax is None: ax = plt.gca()
    if labels is not None: assert len(list_of_results) == len(labels)
    for index, result in enumerate(list_of_results):
        label = result.yaml_config('meta/run_name')
        if labels is not None:
            label = f"{labels[index]}({label})" if labels_add_run_name else labels[index]
        if isinstance(result, Result) and not isinstance(result, MeanStandardResult):
            for i in range(len(result.train_metrics.rewards)):
                label_to_plot = f"{label} - {i}" if add_seed_to_label else label
                ax.plot(np.arange(result.train_metrics.rewards[i].shape[0]), result.train_metrics.rewards[i], label=label_to_plot)
                if label.lower() in ['optimal', 'baseline']: break
        else:
            plot_mean_std(result.train_metrics.rewards[0], result.train_metrics.rewards[1], x=np.arange(result.train_metrics.rewards[0].shape[0]), label=label, ax=ax)
    if have_legend:
        if legend_outside:
            ax.legend(ncol=plt_legend_ncol, bbox_to_anchor=(1, 1), loc='upper left')
        else:
            ax.legend(ncol=plt_legend_ncol)
    ax.set_xlabel(plt_xaxis_label)
    ax.set_ylabel("Reward")
    ax.set_title(plt_title)
    ax.set_yscale(plot_yscale)
    if set_ylim is not None: ax.set_ylim(*set_ylim)
    _a, _b = ax.get_ylim()
    if save_name is not None and folder is not None:
        mysavefig_v2((folder, f'{save_name}.png'));
        plt.close()
    return _a, _b

