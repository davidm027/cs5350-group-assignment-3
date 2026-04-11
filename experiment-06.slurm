#!/bin/bash -l
#SBATCH --job-name=experiment-06
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

mkdir -p results/experiment-06

m=2048
n=4096
q=4096
thread_counts=(1 4 16)

for ((i=0;i<5;i+=1)); do
  for P in "${thread_counts[@]}"; do
    export OMP_NUM_THREADS="$P"
    out="results/experiment-06/dims${m},${n}x${n},${p}_P${P}.csv"

    srun --exclusive ./build/MyAssignment -m $s -n $s -q $s -P $P -s $i >> $filename
    echo "Seed $i: finished $P processors"
  done
  echo "Finished for seed $i"
done

echo "Experiment 6 done!"
