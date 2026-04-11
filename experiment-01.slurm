#!/bin/bash -l
#SBATCH --job-name=experiment-01
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G

set -euo pipefail

CONDA_PREFIX=/apps/anaconda3
module purge
module load anaconda3/2022.05
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate cmake-env

mkdir -p results/experiment-01

P=16
sizes=(512 1024 2048 4096 8192)

for ((i=0;i<5;i+=1)); do
  for s in "${sizes[@]}"; do
    export OMP_NUM_THREADS="$P"
    out="results/experiment-01/m${s}_n${s}_q${s}_P${P}.csv"
    
    srun --exclusive ./build/MyAssignment -m $s -n $s -q $s -P $P -s $i >> $filename
    echo "Seed $i: finished ($s,$s) times ($s,$s)"
  done
  echo "Finished for seed $i"
done

echo "Experiment 1 done!"
