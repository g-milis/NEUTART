#!/bin/bash
set -e


subject=$1
subject_dir=$2


# Train renderer
python photorealistic/renderer/train.py \
    --subject_dir $subject_dir \
    --checkpoints_dir checkpoints/$subject/photorealistic/ \
    --niter 10 \
    --load_pretrain checkpoints/meta-renderer/ \
    --which_epoch 15
