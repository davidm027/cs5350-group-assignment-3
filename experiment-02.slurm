#!/bin/bash -l
#SBATCH --job-name=experiment-02
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

mkdir -p results/experiment-02

P=16
cases=(
  "512 512 1024"
  "1024 1024 2048"
  "2048 2048 4096"
  "4096 4096 8192"
  "512 1024 1024"
  "1024 2048 2048"
  "2048 4096 4096"
  "4096 8192 8192"
  "512 1024 2048"
  "1024 2048 4096"
  "2048 4096 8192"
)

for ((i=0;i<5;i+=1)); do
  for case in "${cases[@]}"; do
    read -r m n q <<< "$case"

    export OMP_NUM_THREADS="$P"
    out="results/experiment-02/m${m}_n${n}_q${q}_P${P}.csv"
    srun --exclusive ./build/MyAssignment -m $s -n $s -q $s -P $P -s $i >> $filename
    echo "Seed $i: finished ($m,$n) times ($n,$q)"
  done
  echo "Finished for seed $i"
done

echo "Experiment 2 done!"
