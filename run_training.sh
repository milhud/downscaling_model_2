#!/bin/bash
#SBATCH --job-name=mrpd_train
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:1           
#SBATCH --ntasks=12            
#SBATCH --mem-per-gpu=100G     
#SBATCH --time=08:00:00        
#SBATCH --account=s1001
#SBATCH --output=training_%j.log
echo -e "\n--- Running Full Training ---"
python3 train.py --batch_size=8 --epochs=50
