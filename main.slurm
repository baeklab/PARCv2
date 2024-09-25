#!/bin/bash
#SBATCH -A sds_baek_energetic
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH -t 23:00:00
#SBATCH --mem=200G

module purge
module load anaconda
source activate gan 

DIR=~/.conda/envs/gan
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH
export PYTHON_PATH=$DIR/lib/python3.7/site-packages
which python
python /home/nmf8dm/research/PARCv2/single_void_refactor.py --name singleVoid_nospade_batchUnit_batchBackbone_diff-snap_resnet50_fe --dir_save /home/nmf8dm/research/PARCv2/reproduce/joseph_resnet56_refactor/differentiator/snap --integrator fe --resnet_blocks 50 --n_ts 1 --epochs 400 --batch_size 8 --version nospade --bn_unit True --bn_backbone True
