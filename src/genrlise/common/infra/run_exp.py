import glob
import os
import subprocess

from genrlise.common.utils import get_date
from genrlise.common.path_utils import path
from genrlise.common.vars import CONDA_ENV_NAME, CONFIG_DIR, LOG_DIR, ROOT_DIR as ROOT_DIR_FUNC, SLURM_DIR, SLURM_LOG_DIR
import fire
from genrlise.common.infra.genrl_config import GenRLExperimentConfig, GenRLSingleRunConfig

def main(
    partition_name: str,
    yaml_config_file: str,
    use_slurm: bool = True, local: bool = None,
    do_log: bool = True,
    override_cores = None
):
    """This creates a slurm file and runs it

    Args:
        partition_name (str): Partition to run the code on
        yaml_config_file (str): The config file to use for everything
        use_slurm (bool): If true, uses slurm, otherwise executes the script with bash
    """
    date = get_date()
    old_filepath = yaml_config_file
    old_fullpath = path('-'.join(yaml_config_file.split("-")[:-1]), yaml_config_file + ".yaml", mk=False)
    if not os.path.exists(yaml_config_file):
        yaml_config_file_new = path(CONFIG_DIR, old_fullpath, mk=False)
        # Only partial name
        if not os.path.exists(yaml_config_file_new):
            print(yaml_config_file_new, 'does not exist')
            yaml_config_file_new = path(CONFIG_DIR, yaml_config_file + ".yaml")
        yaml_config_file = yaml_config_file_new
    if not os.path.exists(yaml_config_file):
        # nested directories
        ans = glob.glob(path(CONFIG_DIR, '*', old_fullpath, mk=False))
        print(path(CONFIG_DIR, '*', old_fullpath, mk=False))
        assert len(ans) == 1
        yaml_config_file = ans[0]
        
    assert os.path.exists(yaml_config_file), f"{yaml_config_file} Does not exist"
    print(yaml_config_file)
    conf = GenRLSingleRunConfig(GenRLExperimentConfig(yaml_config_file), seed='all', date=date)

    assert yaml_config_file.split("/")[-1].replace(".yaml", '') == conf.yaml_config('meta/run_name'), f"{yaml_config_file.split('/')[-1].replace('.yaml', '')} != {conf.yaml_config('meta/run_name')}"

    if local is None: local = not use_slurm
    # hashes = conf.hash(False, False)
    hashes = conf.hash(True, True)
    ROOT_DIR = ROOT_DIR_FUNC(local)
    python_name = conf.yaml_config('meta/file')
    # clean_name = f"{hashes}-{}-{date}"
    # Create Slurm File
    LOG_FILE = f"{ROOT_DIR}/{path(SLURM_LOG_DIR, conf.experiment_name)}/{date}-{hashes}" if not local else f"{path(SLURM_LOG_DIR, conf.experiment_name)}/{date}-{hashes}"
    
    SEED_START = conf.yaml_config('infra/start_seed', 0)
    
    
    how_many_seeds = conf.yaml_config('infra/seeds')
    how_many_cores = conf.yaml_config('infra/cores')
    
    if override_cores is not None:
        print("overriding cores", override_cores)
        how_many_cores = override_cores
        
    li = [f'time ./run.sh {python_name} --is-local {local} --date {date} --overall-name {hashes} --yaml-config-file {yaml_config_file} --do-log {do_log} --particular-seed {MY_SEED} &' for MY_SEED in range(SEED_START, how_many_seeds)]
    if how_many_cores == 4 or how_many_cores == 'auto' and partition_name == 'stampede':
        print("Running 4 experiments at once")
        li.insert(4, 'wait;')
    elif (how_many_cores == 8 and how_many_seeds == 16) and SEED_START == 0:
        print("Running 4 experiments at once")
        li.insert(8, 'wait;')
    elif how_many_cores == 2:
        print("Running 2 experiments at once")
        li.insert(2, 'wait;')
        li.insert(5, 'wait;')
        li.insert(8, 'wait;')
    elif how_many_cores == 3:
        print("Running 2 experiments at once")
        li.insert(3, 'wait;')
        li.insert(7, 'wait;')
    if how_many_cores == 1:
        n = len(li)
        for i in range(n, 0, -1):
            li.insert(i, 'wait;')
    OTHER_EXPORTS = ''
    ALL_RUN_VALUES = '\n'.join(li)
    s = f'''#!/bin/bash
#SBATCH -p {partition_name}
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J {conf.run_name}
#SBATCH -o {LOG_FILE}.%N.%j.out

source ~/.bashrc
cd {ROOT_DIR}
conda activate {CONDA_ENV_NAME}
export WANDB_MODE=disabled
{OTHER_EXPORTS}
echo "{conf.experiment_name} -- with {yaml_config_file} -- {hashes}"
{ALL_RUN_VALUES}
wait;
'''
    dir = path(SLURM_DIR, conf.experiment_name)
    fpath = os.path.join(dir, f'{hashes}.slurm')
    with open(fpath, 'w+') as f:
        f.write(s)

    # Run it    
    if use_slurm:

        ans = subprocess.call(f'sbatch {fpath}'.split(" "))
    else:
        print(f"Logging to {LOG_FILE}, running {fpath}")
        ans = subprocess.Popen(f'bash {fpath} 2>&1'.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ans2 = subprocess.call(f'tee {LOG_FILE}.local.out'.split(" "), stdin=ans.stdout)
        ans.wait()
        ans = ans.returncode
    assert ans == 0
    print("Successfully Ran")
    
if __name__ == '__main__':
    fire.Fire(main)
