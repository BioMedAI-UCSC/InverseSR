#!/bin/bash

#SBATCH --job-name=logHI128
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=vae_g1_HI_128_lr1e-5_%j.out
#SBATCH --error=vae_g1_HI_128_lr1e-5_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

DATASET=HI_128x128x128_stride64
EPOCHS=300
Z_CHANNELS=3
NORM_MODE='global_1'
LR=1e-5
EXPERIMENT_NAME=${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}_${DATASET}_${LR}_NEWOUTLIERS_FILTER

python3 ${SCRATCH}/project/ml_flow_train_vae_decoder_no_augm.py\
    --data_dir "${SCRATCH}/data/inputs/${DATASET}"\
    --tensor_board_logger_vae "${SCRATCH}/logs_vae_paper/vae_decoder_${EXPERIMENT_NAME}"\
    --output_dir_vae "${SCRATCH}/data/trained_model_paper/vae_decoder_${EXPERIMENT_NAME}"\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate $LR\
    --epochs $EPOCHS\
    --batch_size 2\