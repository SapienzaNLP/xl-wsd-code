#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=8000M
# we run on the gpu partition
#SBATCH -p gpu --gres=gpu:titanx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=100:00:00
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo "SLURM_JOBID="$SLURM_JOBID
PYTHONPATH=. python src/training/wsd_trainer.py --config config/config_en_semcor_wngt.train.yaml
