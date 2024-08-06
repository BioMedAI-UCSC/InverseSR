mkdir outputs

echo "$(date +"%T"):  start running model!"

SUBJECT_ID=022
START_STEPS=0
NUM_STEPS=601
LEARNING_RATE=7e-2
LAMBDA_PERC=1e4
CORRUPTION=mask
PRIOR_EVERY=1
PRIOR_AFTER=45
SUBJECT_NUM=7
DATA_FORMAT=nii
MASK_ID=5
DOWNSAMPLE_FACTOR=8
N_SAMPLES=3
K=1
EXPERIMENT_NAME=022_mask_5_draw_img

mkdir result
LOG_DIR=./result/$EXPERIMENT_NAME

python ./project/BRGM_decoder.py \
    --k="$K" \
    --prior_every=$PRIOR_EVERY \
    --prior_after=$PRIOR_AFTER \
    --num_steps="$NUM_STEPS" \
    --data_format="$DATA_FORMAT" \
    --n_samples="$N_SAMPLES" \
    --subject_id="$SUBJECT_ID" \
    --corruption="$CORRUPTION" \
    --mask_id="$MASK_ID" \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name="$EXPERIMENT_NAME" \
    --downsample_factor="$DOWNSAMPLE_FACTOR" \
    --tensor_board_logger="$LOG_DIR"
