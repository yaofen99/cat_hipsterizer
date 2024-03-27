#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --output=test.out
#SBATCH --gres=gpu:v100:1

module purge

module load usc
module load gcc/11.3.0
module load cuda/12.0.0
module load python/3.9.12

pip3 install -r requirements.txt

# python3 preprocess.py
# python3 train.py
python3 preprocess_lmks.py
# python3 train_lmks.py

