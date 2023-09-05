SUBJECT_ID=<> # 069
cp ~/Data/IXI/IXI_T1_unires_pth/IXI_T1_069.pth work/inputs/

cd work
mkdir outputs
# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

START_STEPS=0
NUM_STEPS=601
LAMBDA_PRIOR=0
LEARNING_RATE=7e-2
LAMBDA_PERC=1e4
CORRUPTION=mask
PRIOR_EVERY=15
PRIOR_AFTER=45
DATA_FORMAT=pth
MASK_ID=9
DOWNSAMPLE_FACTOR=8
N_SAMPLES=3
DDIM_NUM_TIMESTEPS=45
K=1
DDIM_ETA=0.0
EXPERIMENT_NAME=069_mask_9_ddim_draw_img
LOG_DIR=/scratch/j/jlevman/jueqi/thesis_experiment_result/$EXPERIMENT_NAME

python3 /scratch/j/jlevman/jueqi/thesis_experiments/ddim/project/BRGM_ddim.py \
    --k=$K \
    --ddim_eta=$DDIM_ETA \
    --ddim_num_timesteps=$DDIM_NUM_TIMESTEPS \
    --update_latent_variables \
    --update_conditioning \
    --mean_latent_vector \
    --update_gender \
    --update_age \
    --update_brain \
    --update_ventricular \
    --prior_every=$PRIOR_EVERY \
    --prior_after=$PRIOR_AFTER \
    --num_steps="$NUM_STEPS" \
    --data_format="$DATA_FORMAT" \
    --n_samples="$N_SAMPLES" \
    --subject_id="$SUBJECT_ID" \
    --corruption="$CORRUPTION" \
    --mask_id="$MASK_ID" \
    --lambda_perc="$LAMBDA_PERC" \
    --lambda_prior="$LAMBDA_PRIOR" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor="$DOWNSAMPLE_FACTOR" \
    --tensor_board_logger="$LOG_DIR"

zip -r /scratch/j/jlevman/jueqi/thesis_experiments/outputs/$EXPERIMENT_NAME.zip outputs/