#!/bin/bash -l
#SBATCH --job-name=experiment-05
#SBATCH --partition=compute
#SBATCH --ntasks=16
#SBATCH --mem=8G

set -euo pipefail

CONDA_PREFIX=/apps/anaconda3
module purge
module load anaconda3/2022.05
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate cmake-env

mkdir -p results_new

m=4096
n=2048
q=4096
thread_counts=(1 4 16)
filename="results_new/experiment-05.csv"
echo "m,n,q,P,MM_ser_sec,MM_1D_sec,MM_2D_sec,1D_speedup,1D_cost,2D_speedup,2D_cost,seed" >> $filename

for ((i=0;i<3;i+=1)); do
  echo "*** Beginning seed $i"
  for P in "${thread_counts[@]}"; do
    echo "seed: $i, m: $m, n: $n, q: $q, P(arg): $P, MPI world_size: 16"
    mpirun -np $P ./build/MyAssignment -m $m -n $n -q $q -P $P -s $i >> $filename
    echo "Seed $i: finished $P processors"
  done
  echo "*** Finished for seed $i"
done

echo "Experiment 5 done!"

