# The options are as follows
# <Experiment> <Slurm Partition> <Do Logging> <Do Run on Slurm> <Is local>
# For instance, ./exp.sh v0190-a batch False False False -> on cluster, run on node without slurm or logging

# If you want to run in parallel, remove `--override-cores=1`
# However, it may fail due to out of memory errors, so you can add it back in.
./run.sh src/genrlise/common/infra/run_exp.py --partition-name ${2:-partition} --use-slurm ${4:-True} --local ${5:-False} --do-log ${3:-False} $1 --override-cores=1

