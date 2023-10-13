from typing import List

import genrlise.rlanalyse as rla
import numpy as np
from common.utils import mysavefig_v2, plot_mean_std
from genrlise.common.vars import RESULTS_DIR
from genrlise.rlanalyse.analyse.main import PRETTY_KWARGS, XLABEL_TRAIN, YLABEL_EVAL, make_legend_good, plot_cartesian_product
from genrlise.rlanalyse.common.result import CHECK_701_SEEDS
from genrlise.common.path_utils import path
from matplotlib import pyplot as plt
ALL_DATA_FOR_STAT_TESTS = {}
PROPER_STRICT_PLOTS = False
DATE='paper_plots_a'

ALL_COLOURS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']

ADAPTER_COLOUR  = ALL_COLOURS[2]
UNAWARE_COLOUR  = ALL_COLOURS[0]
CONCAT_COLOUR   = ALL_COLOURS[1]
FLAP_COLOUR     = ALL_COLOURS[4]
CGATE_COLOUR    = ALL_COLOURS[3]



ADAPTER_NAME = 'Adapter'
CGATE_NAME = 'cGate'
CONCAT_NAME = 'Concat'
UNAWARE_NAME = 'Unaware'
FLAP_NAME = 'FLAP'

def do_all(vals: List[str], 
           labels: List[str], NAME: str,
           max_seeds: int = 100, max_seeds_weights: int= 100, DATE=DATE,
           stop_at_eval_rewards=False, env="ODE",
           plot_kwargs={}, do_mean_standard=True,
           plot_each_seed_alone: bool = False,
           plot_train: bool = False,
           optimal_short_name = None,
           eval_keys=None,
           plot_subplots_force_size=False,
           train_plot_kwargs={},
           read_kwargs={},
           read_in_kwargs={},
           override_state_1=False, do_summary=False,
           do_summary_from_checkpoints=False,
           title_avg=False,
           eval_figsize=None,
           restrict_context=None,
           colours: List[str] = None,
           auc_kwargs={},
           eval_plot_kwargs={}
           ):
    
    save_dir = path(*f'progress/{DATE}/{NAME}/'.split("/"))

    if do_summary:
        rla.plot_eval_summary_distance_to_training_context(vals, labels, save_dir, max_seeds, plot_singles=True, env=env, plot_kwargs=plot_kwargs, do_mean_standard=do_mean_standard, plot_each_seed_alone=plot_each_seed_alone, optimal_short_name=optimal_short_name, eval_keys=eval_keys, plot_subplots_force_size=plot_subplots_force_size, read_kwargs=read_kwargs, override_state_1=override_state_1, read_in_kwargs=read_in_kwargs)
        rla.plot_eval_summary_distance_to_training_context(vals, labels, save_dir, max_seeds, plot_singles=True, env=env, plot_kwargs=plot_kwargs, do_mean_standard=do_mean_standard, plot_each_seed_alone=plot_each_seed_alone, optimal_short_name=optimal_short_name, eval_keys=eval_keys, plot_subplots_force_size=plot_subplots_force_size, read_kwargs=read_kwargs, override_state_1=override_state_1, do_scatter=True, read_in_kwargs=read_in_kwargs)
        model_names = [l.split()[0] for l in labels]
        train_times = [float(l.split()[-1][1:-1].replace("k", '000')) for l in labels]
        rla.plot_eval_summary_auc_vs_train(vals, labels, save_dir, model_names=model_names, train_times=train_times, max_seeds=max_seeds, plot_singles=True, env=env, plot_kwargs=plot_kwargs, do_mean_standard=do_mean_standard, plot_each_seed_alone=plot_each_seed_alone, optimal_short_name=optimal_short_name, eval_keys=eval_keys, plot_subplots_force_size=plot_subplots_force_size, read_kwargs=read_kwargs, override_state_1=override_state_1, eval_figsize=eval_figsize, restrict_context=restrict_context, colours=colours, read_in_kwargs=read_in_kwargs, **auc_kwargs)
        return
    if do_summary_from_checkpoints:
        model_names = labels
        train_times = []
        all_x, all_y, all_std, labels, all_everything = rla.plot_eval_summary_auc_vs_train(vals, labels, save_dir, model_names=model_names, train_times=train_times, max_seeds=max_seeds, plot_singles=True, env=env, plot_kwargs=plot_kwargs, do_mean_standard=do_mean_standard, plot_each_seed_alone=plot_each_seed_alone, optimal_short_name=optimal_short_name, eval_keys=eval_keys, plot_subplots_force_size=plot_subplots_force_size, read_kwargs=read_kwargs, override_state_1=override_state_1, do_checkpoints=True, eval_figsize=eval_figsize, restrict_context=restrict_context, colours=colours, read_in_kwargs=read_in_kwargs, return_all=True, **auc_kwargs)
        return all_x, all_y, all_std, labels, {k: np.array(v).tolist() for k, v in all_everything.items()}
    
    if plot_train:
        rla.plot_train_results_from_list_of_experiments(vals, labels, save_dir, max_seeds, plot_singles=True, env=env, plot_kwargs=plot_kwargs | {'plt_xaxis_label':'Episodes'} | train_plot_kwargs, do_mean_standard=do_mean_standard, plot_each_seed_alone=plot_each_seed_alone, optimal_short_name=optimal_short_name,
                                                        plot_subplots=True, plot_subplots_force_size=plot_subplots_force_size, read_kwargs=read_kwargs, eval_keys=eval_keys)
    
    rla.plot_eval_results_from_list_of_experiments(vals, labels, save_dir, max_seeds, plot_singles=True, env=env, plot_kwargs=plot_kwargs, do_mean_standard=do_mean_standard, plot_each_seed_alone=plot_each_seed_alone, optimal_short_name=optimal_short_name, eval_keys=eval_keys, 
                                                   plot_subplots_force_size=plot_subplots_force_size, read_kwargs=read_kwargs, override_state_1=override_state_1, read_in_kwargs=read_in_kwargs, title_avg=title_avg, eval_figsize=eval_figsize, **eval_plot_kwargs)

def plot_exp(vals, labels, name, env, size=None, eval_override=None, vlines=[], base_size=None, do_summary=False, plot_kwargs={}, **kwargs):
    is_cp = env == 'CartPole' or env == 'XY_CartPole'
    if base_size is not None:
        size = (base_size[0] + 1, base_size[1] + 1)
    if 'ODE' in env:
        name_test = '-10,10'
    else:
        name_test = 'Test{0.1,10}'
    if eval_override is not None: name_test = eval_override
    
    return do_all(
        vals = vals,
        labels=labels,
        NAME=name,
        plot_train=True,
        env=env,
        plot_kwargs=dict(plot_yscale='linear', plt_xaxis_label='Pole Length' if is_cp else 'Context', ax_vlines=vlines,  plt_legend_ncol=2, ) | plot_kwargs,
        eval_keys=[name_test],
        plot_each_seed_alone=True,
        max_seeds=16,
        plot_subplots_force_size=size,
        override_state_1 = env == 'BoundedODE' or env == 'BoundedODE2',
        do_summary=do_summary, **kwargs
    )

#### ODE Main Plots

def ode_v601_also_multidim(do_restrict=True):
    MAX_CHECK = 300_000
    if do_restrict:
        extra = ''
        restrict_context = (-10, 10)
    else:
        extra = '_2020'
        restrict_context = None
    T = []
    vals, labels, colours = zip(*[
        ('v0801a-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
        ('v0801a-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        ('v0801a-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ('v0801a-u3', CGATE_NAME, CGATE_COLOUR),
        ('v0801a-x3', FLAP_NAME, FLAP_COLOUR),
    ])

    ans = plot_exp(vals, labels, f'clean_ode_v0601a{extra}', 'BoundedODE', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), restrict_context=restrict_context, auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK), colours=colours)
    all_x, all_y, all_std, mylabels, ee = ans
    T.append((all_x, all_y, all_std, mylabels))
    
    vals, labels, colours = zip(*[
        ('v0874b-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
        ('v0874b-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        ('v0874b-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ('v0874b-u3', CGATE_NAME, CGATE_COLOUR),
        ('v0874b-z3', FLAP_NAME, FLAP_COLOUR),
    ])
    
    ans = plot_exp(vals, labels, f'ode_multidim_674b{extra}', 'BoundedODE2', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 3), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, ignore_specific_dimensions=True), colours=colours, eval_override='TestCartesianProductProper')

    all_x, all_y, all_std, mylabels, ee = ans
    T.append((all_x, all_y, all_std, mylabels))
    things = ['1D Context', '2D Context']
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for ax, (all_x, all_y, all_std, mylabels), thing in zip(axs, T, things):
        for k, lab, col in zip(all_x.keys(), mylabels, colours):
            plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
            ax.set_title(thing.title())
            if thing != things[-1]: ax.xaxis.get_offset_text().set_visible(False)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fig.supylabel('Mean Eval. Reward')
    for i, a in enumerate(axs): a.hlines([200], 0, 3e5, color='black',  linestyle = 'dashed', label='Optimal')
    save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/ode_single_and_multi_{extra}/'.split("/"))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=6, fontsize=13)
    for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
    fig.supxlabel(XLABEL_TRAIN, y=0.15)
    make_legend_good(leg, skiplast=True)
    plt.tight_layout()
    mysavefig_v2((save_dir, f'both.png'))


#### Appendix Plots

##### ODE Positive/Negative
def ode_v601_also_pos():
    vals, labels, colours = zip(*[
        (f'v0601f-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
        (f'v0601f-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        (f'v0601f-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
    ])

    evalkey = '-10,10'
    
    TT = []
    TT.append(plot_exp(vals, labels, f'test_v0601f_only_pos', 'BoundedODE', base_size=(2, 4), eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, ignore_specific_dimensions=True), colours=colours, eval_override=evalkey, restrict_context=(0, 10), do_summary_from_checkpoints=True))
    
    vals, labels, colours = zip(*[
        (f'v0801a-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
        (f'v0801a-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        (f'v0801a-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
    ])

    evalkey = '-10,10'
    TT.append(plot_exp(vals, labels, f'test_v0801a_pos_and_neg', 'BoundedODE', base_size=(2, 4), eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, ignore_specific_dimensions=True), colours=colours, eval_override=evalkey, restrict_context=(-10, 10), do_summary_from_checkpoints=True))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    PP = ['Positive', 'Positive and Negative']
    pppp = 0
    for ax, (all_x, all_y, all_std, mylabels, _) in zip(axs, TT):
        for k, lab, col in zip(all_x.keys(), mylabels, colours):
            plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
            ax.set_title(PP[pppp])
        pppp += 1
    save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/ode_v0601_compare_pos_all/'.split("/"))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=5, fontsize=16)
    for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
    axs[0].xaxis.get_offset_text().set_visible(False)
    make_legend_good(leg)
    plt.tight_layout()
    axs[0].set_ylabel('Mean Eval. Reward')
    fig.supxlabel(XLABEL_TRAIN)
    mysavefig_v2((save_dir, f'compare_pos_all.png'))

##### CartPole normal Training
def cp_611_clean():
    vals, labels, colours = zip(*[
        ('v0611b-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
        ('v0611b-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        ('v0611b-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ('v0611b-u3', CGATE_NAME, CGATE_COLOUR),
        ('v0611b-z3', FLAP_NAME, FLAP_COLOUR),
    ])

    plot_exp(vals, labels, f'clean_cp_v0611', 'CartPole', size=(1, 6), vlines=[1, 4, 6], do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=900_000))
    plot_exp(vals, labels, f'clean_cp_v0611', 'CartPole', size=(1, 6), vlines=[1, 4, 6])

##### Mass vs Distance Ant

def showcase_ant():
    names = ['Distance Travelled', 'Episode Length', 'Reward']
    for i, read_in_kwargs in enumerate([dict(proper_key_to_use='all_infos'), dict(proper_key_to_use='all_lengths'), dict()]):
        vals, labels, colours = zip(*[
            ('v0527n-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
            ('v0527n-bb3', CONCAT_NAME, CONCAT_COLOUR ),
            ('v0527n-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ])
        
        
        
        plot_exp(vals, labels, f'v0527n_muj_ant/{i}', 'MujocoAnt', base_size=(2, 3), vlines=[1.0], eval_override='TestDensity', do_summary_from_checkpoints=True, eval_figsize=(10, 5), read_in_kwargs=read_in_kwargs, auc_kwargs=dict(legend_under=True), colours=colours)
        plot_exp(vals, labels, f'v0527n_muj_ant/{i}', 'MujocoAnt', base_size=(2, 3), vlines=[5, 35, 75], eval_override='TestDensity', read_in_kwargs=read_in_kwargs, eval_figsize=(10, 5), eval_plot_kwargs=dict(do_pretty_plots=True), plot_kwargs=dict(plt_yaxis_label=names[i], plt_xaxis_label='Mass'))

##### ODE Interpolation/Extrapolation
def ode_v601(do_restrict=True):
    MAX_CHECK = 500_000
    if do_restrict:
        extra = ''
        restrict_context = (-10, 10)
    else:
        extra = '_2020'
        restrict_context = None
    vals, labels, colours = zip(*[
        ('v0801a-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
        ('v0801a-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        ('v0801a-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ('v0801a-u3', CGATE_NAME, CGATE_COLOUR),
        ('v0801a-x3', FLAP_NAME, FLAP_COLOUR),
    ])

    T = []
    ans = plot_exp(vals, labels, f'clean_ode_v0601a{extra}', 'BoundedODE', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 3), restrict_context=restrict_context, auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK), colours=colours)
    all_x, all_y, all_std, mylabels, ee = ans
    T.append((all_x, all_y, all_std, mylabels))
    things = ['train', 'interpolation', 'extrapolation']
    for specific in things:
        all_x, all_y, all_std, mylabels, ee = plot_exp(vals, labels, f'clean_ode_v0601a{extra}', 'BoundedODE', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(7, 3.5), restrict_context=restrict_context, auc_kwargs=dict(specific=specific, have_legend=specific in things[-1], legend_under=False, max_checkpoint=MAX_CHECK))
        T.append((all_x, all_y, all_std, mylabels))

    things.insert(0, 'aggregate')
    plt.close()
    fig = plt.figure(figsize=(10, 5))

    ax1 = plt.subplot(2,3,(1,3))
    ax2 = plt.subplot(2,3,4)
    ax3 = plt.subplot(2,3,5)
    ax4 = plt.subplot(2,3,6)

    
    axs = [ax1, ax2, ax3, ax4]
    for ax, (all_x, all_y, all_std, mylabels), thing in zip(axs, T, things):
        for k, lab, col in zip(all_x.keys(), mylabels, colours):
            plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
            ax.set_title(thing.title())
            if thing == things[2]: ax.set_xlabel(XLABEL_TRAIN)
            if thing != things[-1] and thing != things[0]: ax.xaxis.get_offset_text().set_visible(False)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fig.supylabel('Mean Eval. Reward')
    save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/ode_v0601_interp_extrap{extra}/'.split("/"))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=5, fontsize=16)
    ax3.get_yaxis().set_ticklabels([])
    ax4.get_yaxis().set_ticklabels([])
    for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
    make_legend_good(leg)
    plt.tight_layout()
    mysavefig_v2((save_dir, f'interp_extrap_train.png'))

##### Multidimensional heatmaps
def ode_multidim_674():
    def inner(let, num=3):
        ll = []
        if num == 4 and let == 'b':
            ll = [
                (f'v087{num}{let}-u3', CGATE_NAME, CGATE_COLOUR),
                (f'v087{num}{let}-z3', FLAP_NAME, FLAP_COLOUR),
            ]
        vals, labels, colours = zip(*[
            (f'v087{num}{let}-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
            (f'v087{num}{let}-bb3',  CONCAT_NAME, CONCAT_COLOUR),
            (f'v087{num}{let}-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ] + ll)
        cart_kwargs = dict(first_label='$c_0$', second_label='$c_1$')
        CONVEX, NAME = False, 'NonConvex'
        CONVEX, NAME = 'both', 'Both'
        for j in [1, 6]:
            check = j * 50_000
            read_kwargs=dict(checkpoint=check)
            if j == 6 and (num != 3 or let !='d'):
                read_kwargs=dict(checkpoint=None)
            try:
                plot_cartesian_product(vals, folder=path(RESULTS_DIR, f'progress/{DATE}/ode_67{num}{let}_cartesian/checkpoint_{check}', 'eval_rewards'), key='TestCartesianProductProper', min_rew=0, max_rew=200, read_kwargs=read_kwargs, **cart_kwargs)        
            except:
                pass
        plot_cartesian_product(vals, folder=path(RESULTS_DIR, f'progress/{DATE}/ode_67{num}{let}/{NAME}', 'eval_rewards'), key='TestCartesianProductProper', min_rew=0, max_rew=200, **cart_kwargs, do_convex_hull_marking=CONVEX, do_pairwise_diffs=True)
        for i, evalkey in enumerate(['TestCartesianProductProper']):
            kw = dict(legend_under=True)
            if evalkey != 'TestCartesianProductProper':
                plot_exp(vals, labels, f'ode_67{num}{let}_all/eval_{evalkey}', 'BoundedODE2', size=(1, 6), eval_override=evalkey, colours=colours)
            else:
                kw['ignore_specific_dimensions'] = True
            plot_exp(vals, labels, f'ode_67{num}{let}_all/eval_{evalkey}', 'BoundedODE2', size=(1, 6), eval_override=evalkey, eval_figsize=(10, 5), do_summary_from_checkpoints=True, auc_kwargs=kw, colours=colours)

    if PROPER_STRICT_PLOTS:
        inner('b', 4)
        inner('a', 3)

##### Limitations Appendix Section

###### CartPole Noisy
def clean_cp_noise():
    MAX_CHECK = 600_000
    alls = {
        l: [
            'v0851a-aa3', f'v0851{l}-bb3',f'v0851{l}-fb4',
            f'v0856{l}-bb3',f'v0856{l}-fb4', 
            f'v0857{l}-bb3', f'v0857{l}-fb4',
            
        ] for l in ['a', 'b', 'c', 'd', 'e', 'f']
    }
    tests = {
        'a': 0,
        'b': 0.05,
        'c': 0.1,
        'd': 0.2,
        'e': 0.5,
        'f': 1,
    }
    all_values = []
    my_labels = [UNAWARE_NAME, CONCAT_NAME, ADAPTER_NAME, 
                    CONCAT_NAME + "_NOI", ADAPTER_NAME + "_NOI",
                    CONCAT_NAME + "_NOI2", ADAPTER_NAME + "_NOI2",
                    CONCAT_NAME + "_NOI3", ADAPTER_NAME + "_NOI3",
                    ]
    colours   = [UNAWARE_COLOUR, CONCAT_COLOUR, ADAPTER_COLOUR, 
                    CONCAT_COLOUR, ADAPTER_COLOUR,
                    CONCAT_COLOUR, ADAPTER_COLOUR,
                    CONCAT_COLOUR, ADAPTER_COLOUR
                    ]
    STYLES    = ['-', '-', '-', '--', '--', 'dotted', 'dotted', 'dashdot', 'dashdot']
    STYLES    = ['-', '-', '-', '-', '-', '-', '-', '-', '-', ]
    for k, v in alls.items():
        n = f'clean_cp_651_noise/v0651{k}'
        all_x, all_y, all_std, labels, ee = plot_exp(v, my_labels, n, 'CartPole', base_size=(1, len(v)), do_summary_from_checkpoints=True, auc_kwargs=dict(max_checkpoint=MAX_CHECK))
        all_values.append((all_x, all_y, all_std, labels, tests[k]))

    save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/clean_cp_651_noise/all/'.split("/"))
    all_x, all_y, all_std, labels, _ = all_values[-1]
    fig, axs = plt.subplots(1, 3, figsize=(10, 2), sharey=True)
    for I, (key, label, colour) in enumerate(zip(all_x.keys(), labels, colours)):
        print(label)
        if not label.endswith('NOI') and not label.endswith('NOI2'): 
            ax_index = 0
        elif label.endswith("NOI"):
            ax_index = 1
        else:
            ax_index = 2
        mean, std, xs = [], [], []
        for all_x, all_y, all_std, labels, norm_val in all_values:
            key_to_use = key
            t = all_x[key_to_use][-1]
            y = all_y[key_to_use][-1]
            ss = all_std[key_to_use][-1]
            mean.append(y)
            std.append(ss)
            xs.append(norm_val)
        ii = np.argsort(xs)
        mean = np.array(mean)[ii]
        std = np.array(std)[ii]
        xs = np.array(xs)[ii]
        plot_mean_std(mean, std, x=xs, label=label, color=colour, linestyle=STYLES[I], **PRETTY_KWARGS,ax=axs[ax_index])
        if label.lower() == 'unaware':
            for iii in [1, 2]: plot_mean_std(mean, std, x=xs, label=label, color=colour, linestyle=STYLES[I], **PRETTY_KWARGS,ax=axs[iii])
    handles, labels = axs[0].get_legend_handles_labels()

    import matplotlib.lines as mlines
    a = mlines.Line2D([], [], color='black', ls='-', label='8')
    b = mlines.Line2D([], [], color='black', ls='--', label='8')
    c = mlines.Line2D([], [], color='black', ls='dotted', label='8')

    
    axs[0].set_title("Train Without Noise")
    axs[1].set_title("Train With $\sigma=0.1$")
    axs[2].set_title("Train With $\sigma=0.5$")
    def makes(ll, h=False):
        return ll

    leg = fig.legend(makes(handles, h=True), makes(labels), loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    axs[1].set_xlabel("Level of Context Noise During Evaluation ($\sigma$)")
    axs[0].set_ylabel("Mean Eval. Reward")
    plt.ylim(200, 510)
    mysavefig_v2((save_dir, f'vs_noise_value.png'), pad_inches=0.02)

    pass

###### ODE Norm
def ode_norm_v0601():
    def inner(NUM=1):
        restrict_context, name = None, '_2020'
        restrict_context, name = (-10, 10), ''
        if NUM == 1:
            alls = {
                l: [
                    f'v0801a-aa3',
                    f'v0801{l}-bb3',
                    f'v0801{l}-fb4',
                ] for l in ['a', 'b', 'c', 'd', 'e']
            }
            tests = {
                'a': 5,
                'b': 15,
                'c': 2,
                'd': 1,
                'e': 0.1,
            }
            my_labels = [UNAWARE_NAME, CONCAT_NAME, ADAPTER_NAME]
        else:
            alls = {
                ll: [
                    f'v0803{ll}-bb3',
                    f'v0803{ll}-fb4',
                    f'v0803{ll}-u3',
            ] for ll in ['a', 'b', 'c', 'd', 'e']}
            tests = {
                'a': 0.1,
                'b': 0.5,
                'c': 1,
                'd': 5,
                'e': 15,
            }
            my_labels = [CONCAT_NAME, ADAPTER_NAME, CGATE_NAME]
        all_values = []
        MAX = 300_000
        for k, v in alls.items():
            colours = [UNAWARE_COLOUR, CONCAT_COLOUR, ADAPTER_COLOUR]

            n = f'ode_v060{NUM}_norms/v060{NUM}{k}{name}'
            all_x, all_y, all_std, labels, ee = plot_exp(v, my_labels, n, 'BoundedODE', base_size=(1, len(v)), restrict_context=restrict_context, do_summary_from_checkpoints=True, auc_kwargs=dict(max_checkpoint=MAX))
            all_values.append((all_x, all_y, all_std, labels, tests[k]))
    
        save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/ode_v060{NUM}_norms/all/'.split("/"))
        plt.figure(figsize=(10, 5))
        for key, label, col in zip(all_x.keys(), labels, colours):
            mean, std, xs = [], [], []
            for all_x, all_y, all_std, labels, norm_val in all_values:
                key_to_use = key
                t = all_x[key_to_use][-1]
                y = all_y[key_to_use][-1]
                ss = all_std[key_to_use][-1]
                assert t == MAX or MAX == 0, f"{t=} and {MAX=} {labels=} {key_to_use=} {all_x[key_to_use]}"
                mean.append(y)
                std.append(ss)
                xs.append(norm_val)
            ii = np.argsort(xs)
            mean = np.array(mean)[ii]
            std = np.array(std)[ii]
            xs = np.array(xs)[ii]
            plot_mean_std(mean, std, x=xs, label=label, color=col, **PRETTY_KWARGS)
            leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=5)
            make_legend_good(leg)
            plt.xlabel("Context Normalisation")
            plt.ylabel("Mean Reward over All Evaluation Contexts")
        mysavefig_v2((save_dir, f'vs_norm_value{name}.png'))
    inner(1)

###### CP Norm
def cp_experiments_bad_context_683():
    all_vals = [[
        f'v0981c-aa1',
        f'v0983{l}-bb3',
        f'v0983{l}-fb4',
    ] for l in ['a', 'b']]
    labels = [UNAWARE_NAME, CONCAT_NAME, "Adapter"]
    colours = [UNAWARE_COLOUR, CONCAT_COLOUR, ADAPTER_COLOUR]
    all_names = ['norm_small', 'norm_large']
    answers = {}
    if PROPER_STRICT_PLOTS:
        for vals, name in zip(all_vals, all_names):
            eval_key_to_use = 'Eval(masspole)'
            plot_exp(vals, labels, f'cp_context_bad_683/{name}', 'CartPole', eval_figsize=(10, 5), size=(1, 4), eval_override=eval_key_to_use)
            answers[name] = (plot_exp(vals, labels, f'cp_context_bad_683/{name}', 'CartPole', eval_figsize=(10, 5), size=(1, 4), do_summary_from_checkpoints=True, auc_kwargs=dict(legend_under=True, legend_ncol=6), eval_override=eval_key_to_use, colours=colours))
            
            answers[name] = answers[name][:-1] 
        
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for index, (ax, (context_name)) in enumerate(zip(axs, all_names)):
        all_x, all_y, all_std, mylabels = answers[context_name]
        for k, lab in zip(all_x.keys(), mylabels):
            plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=k, ax=ax, **PRETTY_KWARGS)

            ax.set_title({'norm_small': r"Normalisation$=0.01$", 'norm_large': 'Normalisation$=1$'}[context_name])
            
            if index == 0: ax.set_ylabel(YLABEL_EVAL)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            if index != 1: ax.xaxis.get_offset_text().set_visible(False)
            
            ax.set_ylim(0, 510)
            ax.set_xlim(left=0)
            ax.hlines([500], 0, 1e5, color='black',  linestyle = 'dashed', label='Optimal')
        fig.supxlabel(XLABEL_TRAIN, y=0.05)
    fig = plt.gcf()
    save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/cp_context_bad_683/'.split("/"))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles[::2] + handles[-1:], labels[::2] + labels[-1:], loc='upper center', bbox_to_anchor=(0.5, 0.02), fancybox=True, shadow=True, ncol=7)
    make_legend_good(leg, skiplast=True)
    plt.tight_layout()
    mysavefig_v2((save_dir, f'all_variations.png'), pad_inches=0.02)
    plt.close()

##### Diff. Context Sets
def ode_different_training_settings():
    names = {
        'v0805h': [-0.1, 0.1],
        'v0805c': [-1, -0.1, 0.1, 1],
        
        'v0805b': [1, 5],
        'v0805d': [-5, 5],
        'v0805e': [-5, -4, 4, 5],
        
        'v0805f': [-7.5, 7.5],
        'v0805g': [-7.5, -0.1, 0.1, 7.5],
        'v0801a': [-5, -1, 1, 5],
    }
    vals = labels = [
        'v0801a-aa3',
        'v0801a-bb3',
        'v0801a-fb4',
        
        'v0805a-aa3',
        'v0805a-bb3',
        'v0805a-fb4',
        'v0805b-aa3',
        'v0805b-bb3',
        'v0805b-fb4',
        'v0805c-aa3',
        'v0805c-bb3',
        'v0805c-fb4',
        'v0805d-aa3',
        'v0805d-bb3',
        'v0805d-fb4',
        'v0805e-aa3',
        'v0805e-bb3',
        'v0805e-fb4',
        
        'v0805f-bb3',
        'v0805f-fb4',
        
        'v0805g-aa3',
        'v0805g-bb3',
        'v0805g-fb4',
        'v0805h-aa3',
        'v0805h-bb3',
        'v0805h-fb4',
    ]
    def _f(n):
        a, b = n.split('-')
        A = names[a]
        B = {'bb3': CONCAT_NAME, 'fb4': ADAPTER_NAME}[b]
        return f"{B}({A})"

    ignore = ['v0805a-bb3', 'v0805a-fb4']
    vals = labels = [v for v in vals if '-aa' not in v]
    vals = labels = [v for v in vals if v not in ignore]
    
    RESTR = (-10, 10)

    if PROPER_STRICT_PLOTS:
        all_results = plot_exp(vals, labels, f'ode_605', 'BoundedODE', base_size=(9, 2), do_summary_from_checkpoints=True, restrict_context=RESTR, auc_kwargs=dict(max_checkpoint=300_000))
        all_results = all_results[:-1]
    
    def cleans(li):
        absvals = np.unique([abs(a) for a in li]).tolist()
        ll = []
        for a in absvals:
            if a in li and -a in li:
                ll.append("\pm" + str(a))
            else:
                ll.append(f"{a}")
        
        ll = '[' + ', '.join(ll) + ']'
        L = '$' + ll.replace('[', '\{').replace(']', '\}') + '$'
        print(L)
        return L
    
    fig, axs = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
    for index, (key, ax) in enumerate(zip(names.keys(), axs.ravel())):
        
        all_x, all_y, all_std, mylabels = all_results
        for k, lab in zip(all_x.keys(), mylabels):
            if key not in lab: continue
            if '-bb3' in lab:
                LL, CC = CONCAT_NAME, CONCAT_COLOUR
            if '-fb4' in lab:
                LL, CC = ADAPTER_NAME, ADAPTER_COLOUR
            if '-aa3' in lab:
                LL, CC = UNAWARE_NAME, UNAWARE_COLOUR
            
            plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=LL, ax=ax, color=CC, **PRETTY_KWARGS)
            ax.set_xlabel(f"({chr(97 + index)}): {cleans(names[key])}")
            
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            if index != len(axs.ravel()) - 1: ax.xaxis.get_offset_text().set_visible(False)
            
            ax.set_ylim(0, 200)
            ax.set_xlim(left=0)
        fig.supxlabel(XLABEL_TRAIN, y=0.025)
        fig.supylabel(YLABEL_EVAL)
    fig = plt.gcf()
    save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/ode_trainchange_605/'.split("/"))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles[::2] + handles[-1:], labels[::2] + labels[-1:], loc='upper center', bbox_to_anchor=(0.5, 0.02), fancybox=True, shadow=True, ncol=7)
    make_legend_good(leg)
    plt.tight_layout()
    mysavefig_v2((save_dir, f'all_variations.png'))
    plt.close()



def main_plots():
    global PROPER_STRICT_PLOTS
    PROPER_STRICT_PLOTS = True
    assert CHECK_701_SEEDS
    # This runs all of the paper's plots so we can recreate all plots when changing something.
    ## Main Exp Section
    ### ODE
    ode_v601()
    ode_v601_also_multidim()
    ode_v601_also_pos()
    ode_multidim_674()
    
    ### CP
    ### Noise
    clean_cp_noise()
    ### Proper
    cp_611_clean()
    
    ### Showcase
    showcase_ant()

    ## Failure Cases
    ### ODE Norms
    ode_norm_v0601()
    
    ### CP Norms
    cp_experiments_bad_context_683()
    ### CP Noise
    clean_cp_noise()
    
    ### Training Contexts
    ode_different_training_settings()
    ### Overfit


if __name__ == '__main__':
    main_plots()