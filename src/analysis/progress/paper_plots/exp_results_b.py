import copy
from typing import List

import genrlise.rlanalyse as rla
import numpy as np
from common.utils import mysavefig_v2, plot_mean_std
from genrlise.common.vars import RESULTS_DIR
from genrlise.rlanalyse.analyse.main import PRETTY_KWARGS, XLABEL_TRAIN, make_legend_good
from genrlise.common.path_utils import path
from matplotlib import pyplot as plt
ALL_DATA_FOR_STAT_TESTS = {}
DATE='paper_plots_b'

ALL_COLOURS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']

ADAPTER_COLOUR  = ALL_COLOURS[2]
UNAWARE_COLOUR  = ALL_COLOURS[0]
CONCAT_COLOUR   = ALL_COLOURS[1]
FLAP_COLOUR     = ALL_COLOURS[4]
CGATE_COLOUR    = ALL_COLOURS[3]

COLOURS_DISTRACTOR = [
    '#C97064',
    '#8E5572',
    '#7DAA92',
    '#406E8E',
]


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
        override_state_1 = (env == 'BoundedODE' or env == 'BoundedODE2'), 
        do_summary=do_summary, **kwargs)


def distractor_ode():
    def inner(ENV):
        T = []
        assert ENV == 'ode'
    
        MAX_CHECK = 300_000
        vals, labels, colours = zip(*[
            ('v0801a-bb3', '0',     COLOURS_DISTRACTOR[0]),
            ('v1607_3b-bb1', '1',   COLOURS_DISTRACTOR[1] ),
            ('v1607_3b-bb4', '20',  COLOURS_DISTRACTOR[2] ),
            ('v1607_3b-bb5', '100', COLOURS_DISTRACTOR[3] ),
        ])
        T.append(plot_exp(vals, labels, f'distractor_concat_v1_ode', 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])
        
        vals, labels, colours = zip(*[
            ('v0801a-fb4', '0',     COLOURS_DISTRACTOR[0]),
            ('v1607_3b-fb1', '1',   COLOURS_DISTRACTOR[1] ),
            ('v1607_3b-fb4', '20',  COLOURS_DISTRACTOR[2] ),
            ('v1607_3b-fb5', '100', COLOURS_DISTRACTOR[3] ),
        ])
        T.append(plot_exp(vals, labels, f'distractor_adapter_v1_ode', 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])


        vals, labels, colours = zip(*[
            ('v0801a-u3', '0',     COLOURS_DISTRACTOR[0]),
            ('v1607_3b-u1', '1',   COLOURS_DISTRACTOR[1] ),
            ('v1607_3b-u4', '20',  COLOURS_DISTRACTOR[2] ),
            ('v1607_3b-u5', '100', COLOURS_DISTRACTOR[3] ),
        ])
        plot_exp(vals, labels, f'distractor_cgate_v1_ode', 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1]

            
            
        things = ['Concat', 'Adapter']
        fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        for ax, (all_x, all_y, all_std, mylabels), thing in zip(axs, T, things):
            for k, lab, col in zip(all_x.keys(), mylabels, colours):
                plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
                ax.set_title(thing.title())
                if thing != things[-1]: ax.xaxis.get_offset_text().set_visible(False)

                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        fig.supylabel('Mean Eval. Reward')
        save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/distractor_{ENV}/'.split("/"))
        if ENV == 'cp':
            for a in axs: a.hlines([500], 0, 9e5, color='black',  linestyle = 'dashed', label='Optimal Reward')
        handles, labels = ax.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=5, fontsize=13, title='Number of distractor dimensions')
        for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
        fig.supxlabel(XLABEL_TRAIN, y=0.15)
        make_legend_good(leg, skiplast=True)
        plt.tight_layout()
        mysavefig_v2((save_dir, f'both.png'))
    inner('ode')

def distractor_cp():
    def inner(ENV):
        T = []
        assert ENV == 'cp'
    
        MAX_CHECK = 900_000
        vals, labels, colours = zip(*[
            ('v0611b-bb3', '0',     COLOURS_DISTRACTOR[0]),
            ('v1620_3b-bb1', '1',   COLOURS_DISTRACTOR[1] ),
            ('v1620_3b-bb4', '20',  COLOURS_DISTRACTOR[2] ),
            ('v1620_3b-bb5', '100', COLOURS_DISTRACTOR[3] ),
        ])
        T.append(plot_exp(vals, labels, f'distractor_concat_v1_cp', 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])
        
        vals, labels, colours = zip(*[
            ('v0611b-fb4', '0',     COLOURS_DISTRACTOR[0]),
            ('v1620_3b-fb1', '1',   COLOURS_DISTRACTOR[1] ),
            ('v1620_3b-fb4', '20',  COLOURS_DISTRACTOR[2] ),
            ('v1620_3b-fb5', '100', COLOURS_DISTRACTOR[3] ),
        ])
        T.append(plot_exp(vals, labels, f'distractor_adapter_v1_cp', 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])

                  
        vals, labels, colours = zip(*[
            ('v0611b-u3', '0',     COLOURS_DISTRACTOR[0]),
            ('v1620_3b-u1', '1',   COLOURS_DISTRACTOR[1] ),
            ('v1620_3b-u4', '20',  COLOURS_DISTRACTOR[2] ),
            ('v1620_3b-u5', '100', COLOURS_DISTRACTOR[3] ),
        ])
        plot_exp(vals, labels, f'distractor_cgate_v1_cp', 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1]
  
            
        things = ['Concat', 'Adapter']
        fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        for ax, (all_x, all_y, all_std, mylabels), thing in zip(axs, T, things):
            for k, lab, col in zip(all_x.keys(), mylabels, colours):
                plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
                ax.set_title(thing.title())
                if thing != things[-1]: ax.xaxis.get_offset_text().set_visible(False)

                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        fig.supylabel('Mean Eval. Reward')
        save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/distractor_{ENV}/'.split("/"))
        if ENV == 'cp':
            for a in axs: a.hlines([500], 0, 9e5, color='black',  linestyle = 'dashed', label='Optimal Reward')
        handles, labels = ax.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=5, fontsize=13, title='Number of distractor dimensions')
        for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
        fig.supxlabel(XLABEL_TRAIN, y=0.15)
        make_legend_good(leg, skiplast=True)
        plt.tight_layout()
        mysavefig_v2((save_dir, f'both.png'))
    inner('cp')

def distractor_ant():

    def inner(ENV):
        T = []
        MAX_CHECK = None 
        vals, labels, colours = zip(*[
            ('v0527n-bb3', '0',      COLOURS_DISTRACTOR[0]),
            ('v1788_11a-bb3', '1',   COLOURS_DISTRACTOR[1] ),
            ('v1788_11d-bb3', '20',  COLOURS_DISTRACTOR[2] ),
            ('v1788_11e-bb3', '100', COLOURS_DISTRACTOR[3] ),
        ])
        T.append(plot_exp(vals, labels, f'distractor_concat_v1_ant', 'MujocoAnt', size=(1, 8), do_summary_from_checkpoints=True, eval_figsize=(10, 5), eval_override='TestDensity', auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])
        vals, labels, colours = zip(*[
            ('v0527n-fb4', '0',      COLOURS_DISTRACTOR[0]),
            ('v1788_11a-fb4', '1',   COLOURS_DISTRACTOR[1]  ),
            ('v1788_11d-fb4', '20',  COLOURS_DISTRACTOR[2]  ),
            ('v1788_11e-fb4', '100', COLOURS_DISTRACTOR[3]  ),
        ])
        T.append(plot_exp(vals, labels, f'distractor_adapter_v1_ant', 'MujocoAnt', size=(1, 8), do_summary_from_checkpoints=True, eval_figsize=(10, 5), eval_override='TestDensity', auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])
        
        vals, labels, colours = zip(*[
            ('v0527n-u3', '0',      COLOURS_DISTRACTOR[0]),
            ('v1788_11a-u3', '1',   COLOURS_DISTRACTOR[1]  ),
            ('v1788_11d-u3', '20',  COLOURS_DISTRACTOR[2]  ),
            ('v1788_11e-u3', '100', COLOURS_DISTRACTOR[3]  ),
        ])
        plot_exp(vals, labels, f'distractor_cgate_v1_ant', 'MujocoAnt', size=(1, 8), do_summary_from_checkpoints=True, eval_figsize=(10, 5), eval_override='TestDensity', auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1]
            
            
        things = ['Concat', 'Adapter']
        fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        for ax, (all_x, all_y, all_std, mylabels), thing in zip(axs, T, things):
            for k, lab, col in zip(all_x.keys(), mylabels, colours):
                plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
                ax.set_title(thing.title())
                if thing != things[-1]: ax.xaxis.get_offset_text().set_visible(False)

                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        fig.supylabel('Mean Eval. Reward')
        save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/distractor_{ENV}/'.split("/"))
        handles, labels = ax.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=5, fontsize=13, title='Number of distractor dimensions')
        for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
        fig.supxlabel(XLABEL_TRAIN, y=0.15)
        make_legend_good(leg)
        plt.tight_layout()
        mysavefig_v2((save_dir, f'both.png'))
    inner('ant')

def ode_only_pos():
    vals, labels, colours = zip(*[
        (f'v0601f-aa3', UNAWARE_NAME, UNAWARE_COLOUR),
        (f'v0601f-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        (f'v0601f-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
    ])

    evalkey = '-10,10'
    plot_exp(vals, labels, f'v0601f', 'BoundedODE', base_size=(2, 4), eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, ignore_specific_dimensions=True), colours=colours, eval_override=evalkey, restrict_context=(0, 10))
    


def new_baselines():
    CGATE_EVERY_LAYER_NAME = 'cGateEveryLayer'
    ADAPTER_NO_HNET_NAME   = 'AdapterNoHnet'

    CGATE_EVERY_LAYER_COLOUR    = '#ccb974'
    ADAPTER_NO_HNET_COLOUR      = '#64b5cd'

    MAX_CHECK = 300_000
    extra = ''
    restrict_context = (-10, 10)
 
    T = []
    vals, labels, colours = zip(*[
        ('v0801a-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        ('v0801a-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ('v0801a-u3', CGATE_NAME, CGATE_COLOUR),
        ('v0801a-p3', CGATE_EVERY_LAYER_NAME, CGATE_EVERY_LAYER_COLOUR),
        ('v0801a-s3', ADAPTER_NO_HNET_NAME, ADAPTER_NO_HNET_COLOUR),
    ])

    ans = plot_exp(vals, labels, f'ode_v0601a_other_baselines{extra}', 'BoundedODE', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), restrict_context=restrict_context, auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK), colours=colours)
    all_x, all_y, all_std, mylabels, ee = ans
    T.append((all_x, all_y, all_std, mylabels))

    vals, labels, colours = zip(*[
        ('v0874b-bb3', CONCAT_NAME, CONCAT_COLOUR ),
        ('v0874b-fb4', ADAPTER_NAME, ADAPTER_COLOUR),
        ('v0874b-u3', CGATE_NAME, CGATE_COLOUR),
        ('v0874b-p3', CGATE_EVERY_LAYER_NAME, CGATE_EVERY_LAYER_COLOUR),
        ('v0874b-s3', ADAPTER_NO_HNET_NAME, ADAPTER_NO_HNET_COLOUR),
    ])

    ans = plot_exp(vals, labels, f'ode_multidim_674b_other_baselines{extra}', 'BoundedODE2', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 3), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, ignore_specific_dimensions=True), colours=colours, eval_override='TestCartesianProductProper')
    all_x, all_y, all_std, mylabels, ee = ans
    T.append((all_x, all_y, all_std, mylabels))
    names_of_each = ['1D Context', '2D Context']
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for ax, (all_x, all_y, all_std, mylabels), thing in zip(axs, T, names_of_each):
        for k, lab, col in zip(all_x.keys(), mylabels, colours):
            plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
            ax.set_title(thing.title())
            if thing != names_of_each[-1]: ax.xaxis.get_offset_text().set_visible(False)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fig.supylabel('Mean Eval. Reward')
    for i, a in enumerate(axs): a.hlines([200], 0, 3e5, color='black',  linestyle = 'dashed', label='Optimal')
    save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/ode_single_and_multi_other_baselines{extra}/'.split("/"))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=6, fontsize=13)
    for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
    fig.supxlabel(XLABEL_TRAIN, y=0.15)
    make_legend_good(leg, skiplast=True)
    plt.tight_layout()
    mysavefig_v2((save_dir, f'both.png'))

def adapter_ablations():
    RESTRICT = (-10, 10)
    EX = ''

    I = True
    FIGSIZE = (10, 5)
    if I:
        vals, labels = zip(*[
            ('v0861a-fc10', 'Act after trunk and features'),
            ('v0861a-fc3', 'Normal Features'),
            ('v0861e-fb4', 'Reverse state and context'),
        ])
        
        plot_exp(vals, labels, f'ode_v0661a{EX}/features_and_activation', 'BoundedODE', eval_figsize=FIGSIZE, size=(1, 4), title_avg=True)
        
        plot_exp(vals, labels, f'ode_v0661a{EX}/features_and_activation', 'BoundedODE', eval_figsize=FIGSIZE, size=(1, 4), do_summary_from_checkpoints=True, auc_kwargs=dict(legend_under=True, max_checkpoint=300_000), restrict_context=RESTRICT)
    if I:
        vals, labels = zip(*[
            ('v0801a-fb4', 'Skip Connection'),
            ('v0861b-fb1', 'No Skip Connection'),
        ])
        
        plot_exp(vals, labels, f'ode_v0661a{EX}/skip_vs_not', 'BoundedODE', eval_figsize=FIGSIZE, size=(1, 3), title_avg=True)
        plot_exp(vals, labels, f'ode_v0661a{EX}/skip_vs_not', 'BoundedODE', eval_figsize=FIGSIZE, size=(1, 3), do_summary_from_checkpoints=True, auc_kwargs=dict(legend_under=True, max_checkpoint=300_000), restrict_context=RESTRICT)
    if I:
        vals, labels = zip(*[
            ('v0861b-fb2', r'$()$'),        
            ('v0861b-fb3', r'$(8)$'),       
            ('v0861b-fb8', r'$(16)$'),      
            ('v0801a-fb4', r'$(32)$'),      
            ('v0861b-fb7', r'$(64)$'),      
            ('v0861b-fb4', r'$(256)$'),     
            ('v0861b-fb5', r'$(32, 32)$'),  
        ])
        labels = [l.replace("(", '[').replace(")", ']') for l in labels]
        
        plot_exp(vals, labels, f'ode_v0661a{EX}/adapter_arch', 'BoundedODE', eval_figsize=FIGSIZE, size=(1, 8), do_summary_from_checkpoints=True, auc_kwargs=dict(legend_under=True, max_checkpoint=300_000, legend_title='Adapter Architecture', legend_ncol=7), restrict_context=RESTRICT)
    if I:
        vals, labels = zip(*[
            ('v0801a-fb4', '${Base}$',),
            ('v0861a-fc3', '${Base}_F$',),
            
            ('v0861a-fc1', r'${Start}$'),
            ('v0861b-fb11', '$End$',),
            
            ('v0861a-fc7', r'${All}_F$'),

        ])
        
        plot_exp(vals, labels, f'ode_v0661a{EX}/adapter_location', 'BoundedODE', size=(1, 7), do_summary_from_checkpoints=True, auc_kwargs=dict(legend_under=True, max_checkpoint=300_000), restrict_context=RESTRICT, eval_figsize=(10, 5))
    if I:
        vals, labels = zip(*[
            ('v0801a-fb4', '${Base}$',),
            ('v0861d-fb1', '[]',),
            ('v0861d-fb2', '[100,100]',),
            ('v0861d-fb3', '[100,100,100]',),
            ('v0861d-fb5', 'Chunked [66, 66]',),
        ])
        labels = [l.replace("(", '[').replace(")", ']') for l in labels]

        plot_exp(vals, labels, f'ode_v0661d{EX}/hnets', 'BoundedODE', eval_figsize=FIGSIZE, size=(1, 5), do_summary_from_checkpoints=True, auc_kwargs=dict(legend_under=True, max_checkpoint=300_000, legend_title='Hypernetwork Architecture', legend_ncol=7, leg_fontsize=8), restrict_context=RESTRICT)
        plot_exp(vals, labels, f'ode_v0661d{EX}/hnets', 'BoundedODE', eval_figsize=FIGSIZE, size=(1, 5), restrict_context=RESTRICT)
    if I:
        vals, labels = zip(*[
            ('v0801a-fb4', r'No Activation'),
            ('v0861f-fd1', r'No Activation (Critic)'),
            ('v0801a-fb3', r'Activation'),
            ('v0801a-fb5', r'Only $\log(\sigma)$ Activation'),
        ])
        
        plot_exp(vals, labels, f'ode_v0661a{EX}/adapter_act_end', 'BoundedODE', size=(1, 4), eval_figsize=FIGSIZE, do_summary_from_checkpoints=True, auc_kwargs=dict(legend_under=True, max_checkpoint=300_000), restrict_context=RESTRICT)



def gaussian_distractors():
    def inner(ENV, n='b', which='normal'):
        T = []
        SIGMA = 0.2
        MYNUM = 3
        MAX_CHECK = 300_000
        
        if which == 'normal':
            NAME_1 = f'v2_new_gaussian_distractor_{SIGMA}_concat_v1_{ENV}'
            NAME_2 = f'v2_new_gaussian_distractor_{SIGMA}_adapter_v1_{ENV}'
            NAME_3 = f'v2_new_gaussian_distractor_{SIGMA}_cgate_v1_{ENV}'
        elif which == 'same_train_test_0':
            MAX_CHECK = 300_000
            MYNUM = 9
            NAME_1 = f'v2_new_gaussian_distractor_{SIGMA}_concat_v1_{ENV}_same_train_test_0'
            NAME_2 = f'v2_new_gaussian_distractor_{SIGMA}_adapter_v1_{ENV}_same_train_test_0'
            NAME_3 = f'v2_new_gaussian_distractor_{SIGMA}_cgate_v1_{ENV}_same_train_test_0'
        elif which == 'same_train_test_1':
            MYNUM = 2
            NAME_1 = f'v2_new_gaussian_distractor_{SIGMA}_concat_v1_{ENV}_same_train_test_1'
            NAME_2 = f'v2_new_gaussian_distractor_{SIGMA}_adapter_v1_{ENV}_same_train_test_1'
            NAME_3 = f'v2_new_gaussian_distractor_{SIGMA}_cgate_v1_{ENV}_same_train_test_1'
        
        if ENV == 'ode':
            vals, labels, colours = zip(*[
                ('v0801a-bb3', '0',      COLOURS_DISTRACTOR[0]),
                (f'v1608_{MYNUM}{n}-bb1', '1',    COLOURS_DISTRACTOR[1] ),
                (f'v1608_{MYNUM}{n}-bb4', '20',   COLOURS_DISTRACTOR[2] ),
                (f'v1608_{MYNUM}{n}-bb5', '100',  COLOURS_DISTRACTOR[3] ),
            ])
            if n == 'a':
                NAME_1 = f'new_gaussian_distractor_{SIGMA}_concat_v1_ode_distractor_always_0'
                NAME_2 = f'new_gaussian_distractor_{SIGMA}_adapter_v1_ode_distractor_always_0'
            T.append(plot_exp(vals, labels, NAME_1, 'BoundedODE', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6, do_ylim=(0, 200)), colours=colours, restrict_context=(-10, 10))[:-1])
        
            vals, labels, colours = zip(*[
                ('v0801a-fb4', '0',     COLOURS_DISTRACTOR[0]),
                (f'v1608_{MYNUM}{n}-fb1', '1',   COLOURS_DISTRACTOR[1]),
                (f'v1608_{MYNUM}{n}-fb4', '20',  COLOURS_DISTRACTOR[2]),
                (f'v1608_{MYNUM}{n}-fb5', '100', COLOURS_DISTRACTOR[3]),
            ])
            T.append(plot_exp(vals, labels, NAME_2, 'BoundedODE', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6, do_ylim=(0, 200)), colours=colours, restrict_context=(-10, 10))[:-1])
            
            
            vals, labels, colours = zip(*[
                ('v0801a-u3', '0',     COLOURS_DISTRACTOR[0]),
                (f'v1608_{MYNUM}{n}-u1', '1',   COLOURS_DISTRACTOR[1]),
                (f'v1608_{MYNUM}{n}-u4', '20',  COLOURS_DISTRACTOR[2]),
                (f'v1608_{MYNUM}{n}-u5', '100', COLOURS_DISTRACTOR[3]),
            ])
            (plot_exp(vals, labels, NAME_3, 'BoundedODE', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6, do_ylim=(0, 200)), colours=colours, restrict_context=(-10, 10))[:-1])
            
            
            if n == 'a': return
        else:
            vals, labels, colours = zip(*[
                ('v0611b-bb3', '0',     COLOURS_DISTRACTOR[0]),
                (f'v1609_{MYNUM}{n}-bb1', '1',   COLOURS_DISTRACTOR[1] ),
                (f'v1609_{MYNUM}{n}-bb4', '20',  COLOURS_DISTRACTOR[2] ),
                (f'v1609_{MYNUM}{n}-bb5', '100', COLOURS_DISTRACTOR[3] ),
            ])
            T.append(plot_exp(vals, labels, NAME_1, 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])
            
            vals, labels, colours = zip(*[
                ('v0611b-fb4', '0',     COLOURS_DISTRACTOR[0]),
                (f'v1609_{MYNUM}{n}-fb1', '1',   COLOURS_DISTRACTOR[1] ),
                (f'v1609_{MYNUM}{n}-fb4', '20',  COLOURS_DISTRACTOR[2] ),
                (f'v1609_{MYNUM}{n}-fb5', '100', COLOURS_DISTRACTOR[3] ),
            ])
            T.append(plot_exp(vals, labels, NAME_2, 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])


            vals, labels, colours = zip(*[
                ('v0611b-u3', '0',     COLOURS_DISTRACTOR[0]),
                (f'v1609_{MYNUM}{n}-u1', '1',   COLOURS_DISTRACTOR[1]),
                (f'v1609_{MYNUM}{n}-u4', '20',  COLOURS_DISTRACTOR[2]),
                (f'v1609_{MYNUM}{n}-u5', '100', COLOURS_DISTRACTOR[3]),
            ])
            (plot_exp(vals, labels, NAME_3, 'CartPole', size=(1, 6), do_summary_from_checkpoints=True, eval_figsize=(10, 5), auc_kwargs=dict(legend_under=True, max_checkpoint=MAX_CHECK, legend_ncol=6), colours=colours)[:-1])
            
            
        things = ['Concat', 'Adapter']
        fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        for ax, (all_x, all_y, all_std, mylabels), thing in zip(axs, T, things):
            for k, lab, col in zip(all_x.keys(), mylabels, colours):
                plot_mean_std(all_y[k], all_std[k], x=all_x[k], label=f"{k}", ax=ax, color=col, **PRETTY_KWARGS)
                ax.set_title(thing.title())
                if thing != things[-1]: ax.xaxis.get_offset_text().set_visible(False)

                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        fig.supylabel('Mean Eval. Reward')
        save_dir = path(*f'{RESULTS_DIR}/progress/{DATE}/v2_new_gaussian_distractor_{which}_{SIGMA}_{ENV}/'.split("/"))
        if ENV == 'cp':
            for a in axs: a.hlines([500], 0, 3e5, color='black',  linestyle = 'dashed', label='Optimal Reward')
        handles, labels = ax.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=5, fontsize=13, title='Number of distractor dimensions')
        for a in axs[1:]: a.set_ylim(axs[1].get_ylim())
        fig.supxlabel(XLABEL_TRAIN, y=0.15)
        make_legend_good(leg, skiplast=True)
        plt.tight_layout()
        mysavefig_v2((save_dir, f'both.png'))

    inner('ode', 'b', 'normal')
    inner('cp', 'b' , 'normal')
    
    # No Change between training and testing
    inner('ode', 'b', 'same_train_test_0')
    inner('cp', 'b' , 'same_train_test_0')
    inner('ode', 'b', 'same_train_test_1')
    inner('cp', 'b' , 'same_train_test_1')



if __name__ == '__main__':
    distractor_ant()
    distractor_cp()
    distractor_ode()
    ode_only_pos()
    new_baselines()
    adapter_ablations()
    gaussian_distractors()