#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --account=p_discoret
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=70G  
#SBATCH --partition=alpha
module load modenv/hiera GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 Python/3.8.6 PyTorch/1.9.0
source ~/python-environments/torchvision_env/bin/activate
python df_0.50.py 

