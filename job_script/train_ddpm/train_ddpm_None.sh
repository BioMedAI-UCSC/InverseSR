#!/bin/bash

#SBATCH --job-name=g1ddpm_None
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=ddpm_None_HI_g1_Vae300_lr1e-5_%j.out
#SBATCH --error=ddpm_None_HI_g1_Vae300_lr1e-5_%j.err
#SBATCH --mem=160G

module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate


DATASET=HI_128x128x128_stride64
EPOCHS=400
Z_CHANNELS=3
NORM_MODE='global_1'
COND='None'
LR=1e-5
EXPERIMENT_NAME=${COND}_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}_${DATASET}_${LR}_NEWFILTER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 ${SCRATCH}/project/ml_flow_train_ddpm_v3_no_augm.py\
    --data_dir "${SCRATCH}/data/inputs/${DATASET}"\
    --in_channels_unet $Z_CHANNELS\
    --out_channels_unet $Z_CHANNELS\
    --tensor_board_logger_ddpm "${SCRATCH}/logs_ddpm_paper/ddpm_${EXPERIMENT_NAME}"\
    --output_dir_ddpm "${SCRATCH}/data/trained_model_paper/ddpm_${EXPERIMENT_NAME}"\
    --epochs $EPOCHS\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate $LR\
    --cond_key $COND\
    --vae_path "${SCRATCH}/vae_decoder_HI_128x128x128_stride64_3_300epochs_global_1_Nov18_15-58-29/vae_best_model.pth"\
    --no-use_mask_channel \
    --no-use_spatial_transformer \
    --batch_size 2\
