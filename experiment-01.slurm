#!/bin/bash -l
#SBATCH --job-name=experiment-01
#SBATCH --partition=compute
#SBATCH --mem=12G
#SBATCH --ntasks=16

set -euo pipefail

CONDA_PREFIX=/apps/anaconda3
module purge
module load anaconda3/2022.05
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate cmake-env

mkdir -p results_new

P=16
sizes=(512 1024 2048 4096)
filename="results_new/experiment-01.csv"
echo "m,n,q,P,MM_ser_sec,MM_1D_sec,MM_2D_sec,1D_speedup,1D_cost,2D_speedup,2D_cost,seed" >> $filename

for ((i=0;i<3;i+=1)); do
  echo "*** Beginning seed $i"
  for s in "${sizes[@]}"; do
    echo "seed: $i, m: $s, n: $s, q: $s, P(arg): $P, MPI world_size: 16"
    mpirun -np $P ./build/MyAssignment -m $s -n $s -q $s -P $P -s $i >> $filename
    echo "Seed $i: finished ($s,$s) times ($s,$s)"
  done
  echo "*** Finished for seed $i"
done

echo "Experiment 1 done!"

