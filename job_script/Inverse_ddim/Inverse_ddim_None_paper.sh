#!/bin/bash

#SBATCH --job-name=Inv_ddim
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=Inv_ddim_g1_HIskyldev_newDown_%j.out
#SBATCH --error=Inv_ddim_g1_HIskyldev_newDown_%j.err
#SBATCH --mem=160G




module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

DATASET=HI_skyldev_128x128x128_stride64
VAE_A="${SCRATCH}/data/trained_model_paper/vae_decoder_3_300epochs_global_arcsinh_Oct10_20-57-08_best_val/VAE_full_best"
DDPM_A="${SCRATCH}/data/trained_model_paper/ddpm_None_1000_z3_global_arcsinh_16x128x128_cont_ldev_OK_augm_vae_best/ddpm_best_model"
DDPM_R="${SCRATCH}/data/trained_model_paper/ddpm_None_1000_z3_global_robust_16x128x128_cont_ldev_OK_augm_vae_best/ddpm_best_model"
VAE_R="${SCRATCH}/data/trained_model_paper/vae_decoder_3_300epochs_global_robust_Oct10_20-55-53_best_val/VAE_full_best"
VAE_L="${SCRATCH}/data/trained_model_paper/vae_decoder_3_300epochs_local_augm_Oct10_20-58-21_best_val/VAE_full_best"
DDPM_L="${SCRATCH}/data/trained_model_paper/ddpm_None_1000_z3_local_16x128x128_cont_ldev_OK_augm_vae_best/ddpm_best_model"
VAE_G2="${SCRATCH}/data/trained_model_paper/vae_decoder_3_300epochs_global_Oct15_11-52-06_best_val/VAE_full_best"
DDPM_G2="${SCRATCH}/data/trained_model_paper/ddpm_None_500_z3_global_16x128x128_cont_ldev_OK_augm_vae_best/ddpm_best_model"
VAE_G1="${SCRATCH}/data/trained_model_paper/vae_decoder_300_z3_global_1_16x128x128_cont_ldev_OK_val_augm/vae_best_model"
DDPM_G1="${SCRATCH}/data/trained_model_paper/ddpm_None_500_z3_global_1_16x128x128_cont_ldev_OK_augm_vae_best/ddpm_best_model"

DDPM_G1_NEW="${SCRATCH}/data/trained_model_paper/ddpm_None_50_z3_global_1_HI_128x128x128_stride64_vae_best/ddpm_best_model"
VAE_G1_NEW="${SCRATCH}/data/trained_model_paper/vae_decoder_100_z3_global_1_HI_128x128x128_stride64_newActivationFun/vae_best_model"

VAE_SKY_G1="${SCRATCH}/VAE_SKYLDEV_G1_MLFLOW"
DDPM_SKY_G1="${SCRATCH}/DDPM_SKYLDEV_G1_MLFLOW"

VAE_LOG="${SCRATCH}/data/trained_model_paper/vae_decoder_100_z3_log_minmax_HI_128x128x128_stride64_newActivationFun/vae_best_model"
DDPM_LOG="${SCRATCH}/data/trained_model_paper/ddpm_None_50_z3_log_minmax_HI_128x128x128_stride64_vae_best/ddpm_best_model"
NORM_DATA='global_1'
LAMBDA_PRIOR=0.08
LEARNING_RATE=0.04
LAMBDA_PERC=0
CORRUPTION=downsample
PRIOR_EVERY=1
DATA_FORMAT="npy"
DOWNSAMPLE_FACTOR=4
DDIM_ETA=0.0
COND_KEY="None"
NUM_STEPS=500
DDIM_NUM_STEPS=45
DELTA=0.001
PATIENCE=15
EXPERIMENT_NAME=NewDown_${DATASET}_lambda${LAMBDA_PERC}_${NORM_DATA}_ddimsteps${DDIM_NUM_STEPS}_lr${LEARNING_RATE}_lp${LAMBDA_PRIOR}_pe${PRIOR_EVERY}_pat${PATIENCE}_${DELTA}_409
Z_CHANNELS=3
LOG_DIR=$SCRATCH/logs_paper/$EXPERIMENT_NAME



export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

python3 ${SCRATCH}/project/BRGM_ddim_cond_v2.py \
    --hr_path ${SCRATCH}/data/inputs/${DATASET}/test/npy_patches/patch_000004_sky_ldev.npy\
    --lr_path ${SCRATCH}/data/inputs/${DATASET}/test_LR/patch_000004_sky_ldev.npy\
    --patience $PATIENCE \
    --min_delta $DELTA \
    --image_size 128 128 128 \
    --inference \
    --z_channels $Z_CHANNELS \
    --cond_key $COND_KEY\
    --out_channels 1\
    --ddim_num_timesteps $DDIM_NUM_STEPS\
    --num_steps $NUM_STEPS\
    --ddim_eta $DDIM_ETA \
    --update_latent_variables \
    --mean_latent_vector \
    --prior_every $PRIOR_EVERY \
    --lambda_prior $LAMBDA_PRIOR\
    --data_format=$DATA_FORMAT \
    --corruption=$CORRUPTION \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor $DOWNSAMPLE_FACTOR \
    --tensor_board_logger_ddim="$LOG_DIR" \
    --output_dir_BRGM_ddim="${SCRATCH}/data/outputs_paper/BRGM_ddim_${EXPERIMENT_NAME}" \
    --norm_data $NORM_DATA \
    --vae_path_BRGM $VAE_SKY_G1\
    --ddpm_path_BRGM $DDPM_SKY_G1 \
    
    

