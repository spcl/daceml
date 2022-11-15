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

outfile=./out.csv
for hidden in 16 128 512; do
  echo python benchmark.py --hidden $hidden --outfile $outfile --model gat
  python benchmark.py --hidden $hidden --outfile $outfile --model gat
done
