#!/bin/bash -l
#SBATCH --job-name=experiment-02
#SBATCH --partition=compute
#SBATCH --ntasks=16
#SBATCH --mem=12G

set -euo pipefail

CONDA_PREFIX=/apps/anaconda3
module purge
module load anaconda3/2022.05
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate cmake-env

mkdir -p results_new

P=16
cases=(
  "512 512 1024"
  "1024 1024 2048"
  "2048 2048 4096"
  "512 1024 1024"
  "1024 2048 2048"
  "2048 4096 4096"
  "512 1024 2048"
  "1024 2048 4096"
)
filename="results_new/experiment-02.csv"
echo "m,n,q,P,MM_ser_sec,MM_1D_sec,MM_2D_sec,1D_speedup,1D_cost,2D_speedup,2D_cost,seed" >> $filename

for ((i=0;i<3;i+=1)); do
  echo "*** Beginning seed $i"
  for case in "${cases[@]}"; do
    read -r m n q <<< "$case"
    echo "seed: $i, m: $m, n: $n, q: $q, P(arg): $P, MPI world_size: 16"
    mpirun -np $P ./build/MyAssignment -m $m -n $n -q $q -P $P -s $i >> $filename
    echo "Seed $i: finished ($m,$n) times ($n,$q)"
  done
  echo "*** Finished for seed $i"
done

echo "Experiment 2 done!"

