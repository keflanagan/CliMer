#!/bin/bash

echo $1

if [ "$1" == "ego4d" ]; then
    source config/ego4d.config
    echo 'ego4d'

elif [ "$1" == "epic" ]; then
    source config/epic.config
    echo 'epic'
fi

python train_test/evaluation.py \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --video-feature-path $VIDEO_FEATURE_PATH \
    --caption-data-train $CAPTION_DATA_TRAIN \
    --caption-data-val $CAPTION_DATA_VAL \
    --caption-data-test $CAPTION_DATA_TEST \
    --ego4d-metadata $EGO4D_METADATA \
    --epic-metadata $EPIC_METADATA \
    --epic-all-data $EPIC_ALL_DATA \
    --bert-features $BERT_FEATURES \
    --results-path $RESULTS_PATH \
    --egovlp-data $EGOVLP_DATA \
    --log-dir $LOG_DIR \
    --visual-feature-orig-dim $VISUAL_FEATURE_ORIG_DIM \
    --learning-rate $LEARNING_RATE \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --epochs $EPOCHS \
    --val-frequency $VAL_FREQUENCY \
    --log-frequency $LOG_FREQUENCY \
    --checkpoint-save-path $CHECKPOINT_SAVE_PATH \
    --worker-count $WORKER_COUNT \
    --same-vid-sampling $SAME_VID_SAMPLING \
    --cross-attention $CROSS_ATTENTION \
    --balanced $BALANCED \
    --combine $COMBINE \
    --fixed-clip-length $FIXED_CLIP_LENGTH \
    --clip-adjacent-timestamps $CLIP_ADJACENT_TIMESTAMPS \
    --egovlp $EGOVLP \
    --checkpoint-save-path $CHECKPOINT_SAVE_PATH \
    --checkpoint-frequency $CHECKPOINT_FREQUENCY \
    --fps $FPS \
    --feature-stride $FEATURE_STRIDE \
    --seg-size $SEG_SIZE \
    --overlap $OVERLAP \
    --pred-threshold $PRED_THRESHOLD \
    --visual-feature-orig-dim $VISUAL_FEATURE_ORIG_DIM \
    --cap-orig-dim $CAP_ORIG_DIM \
    --shared-projection-dim $SHARED_PROJECTION_DIM \
    --feature-embed-dim $FEATURE_EMBED_DIM \
    --linear-hidden-dim $LINEAR_HIDDEN_DIM \
    --num-heads $NUM_HEADS

