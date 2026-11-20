#!/bin/bash

#SBATCH --job-name=test_ddpm
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=test_ddpm_skyldev_%j.out
#SBATCH --error=test_ddpm_skyldev_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7
SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

python3 ${SCRATCH}/project/test_ddpm.py\
    --test_dir "$SCRATCH/data/inputs/HI_skyldev_128x128x128_stride64/test/npy_patches"\
    --in_channels_unet 3\
    --out_channels_unet 3\
    --z_channels 3\
    --norm_mode global_1\
    --cond_key None\
    --vae_path  $SCRATCH/vae_decoder_HI_skyldev_128x128x128_stride64_3_100epochs_global_1_Nov06_15-56-46/vae_best_model.pth\
    --output_dir_ddpm $SCRATCH/ddpm_None_HI_skyldev_128x128x128_stride64_400epochs_global_1_Nov11_12-42-03/ddpm_best_epoch.pth\
    --test_fig $SCRATCH/data/outputs_paper/test_ddpm_skyldev_global1\
    --no-use_mask_channel \
    --no-use_spatial_transformer \
    --image_size 128 128 128 \
    --batch_size 2\
    