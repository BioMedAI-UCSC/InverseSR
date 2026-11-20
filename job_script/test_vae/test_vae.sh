#!/bin/bash

#SBATCH --job-name=test_vae
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=test_vae_glob1_HI_skyldev_%j.out
#SBATCH --error=test_vae_glob1_HI_skyldev_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate
NORM_MODE="global_1"

python3 ${SCRATCH}/project/test_vae.py\
    --test_dir "$SCRATCH/data/inputs/HI_skyldev_128x128x128_stride64/test/npy_patches"\
    --z_channels 3\
    --norm_mode $NORM_MODE\
    --vae_path  "$SCRATCH/vae_decoder_HI_skyldev_128x128x128_stride64_3_100epochs_global_1_Nov06_15-56-46/vae_best_model.pth"\
    --test_fig "$SCRATCH/data/outputs_paper/test_vae_HI_skyldev_${NORM_MODE}_Nov06_best"