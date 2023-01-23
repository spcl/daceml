#!/bin/bash
#SBATCH --job-name=gnn_benchmark      # Job name
#SBATCH --time=00:30:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=gnn_benchmark_%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH -p intelv100
#SBATCH --account=g34
#SBATCH --gpus=1


echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

source /users/jbazinsk/miniconda3/bin/activate
conda activate env
module load cuda/11.4.0
export LIBRARY_PATH=/users/jbazinsk/miniconda3/envs/env/lib/:$LIBRARY_PATH

rm -rf ./.dacecache

do_test=

outfile=./out-$(hostname -s)-$(date +%d.%m.%H.%M)-$SLURM_JOB_ID.csv
## GAT
model=gat
for hidden in 8 32 128 512; do
  echo "Hidden" $hidden
  rm -rf .dacecache
  $do_test python benchmark.py --hidden $hidden --outfile $outfile --model $model --opt
  rm -rf .dacecache
  $do_test python benchmark.py --hidden $hidden --outfile $outfile --model $model --persistent-mem --threadblock-dynamic --opt
done

## GCN
model=gcn
for hidden in 8 32 128 512; do
  echo "Hidden" $hidden
  rm -rf .dacecache
  $do_test python benchmark.py --hidden $hidden --outfile $outfile --model $model --opt
  rm -rf .dacecache
  $do_test python benchmark.py --hidden $hidden --outfile $outfile --model $model --persistent-mem --opt --threadblock-dynamic
done
