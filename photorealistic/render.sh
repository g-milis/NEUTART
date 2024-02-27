#!/bin/bash
set -e


subject=$1
name=$2


# The paths where the audio and the 3D talking head parameters are saved
pth_file=output/$subject/$name.pth
wav_file=output/$subject/$name.wav

# Directory where the reference video is preprocessed
subject_dir=processed_videos/${subject}_test
# Directory where the rendered checkpoints are saved
checkpoints_dir=checkpoints/$subject/photorealistic


if ! [ -d $subject_dir/$name/SPECTRE ]; then 
    mkdir -p $subject_dir/$name/SPECTRE
    cp $subject_dir/SPECTRE/* $subject_dir/$name/SPECTRE
fi

# Creates shape and NMFC images (renderer inputs)
python photorealistic/renderer/create_inputs.py \
    --subject_dir $subject_dir \
    --exp_name $name \
    --input $pth_file

# Creates aligned faces
python photorealistic/renderer/test.py \
    --subject_dir $subject_dir \
    --exp_name $name \
    --checkpoints_dir $checkpoints_dir

# Unaligns faces and writes video frames
python photorealistic/postprocessing/postprocess.py \
    --subject_dir $subject_dir \
    --exp_name $name

# Create mp4 from video frames
python photorealistic/postprocessing/images2video.py \
    --imgs_path $subject_dir/$name/full_frames \
    --out_path output/$subject/$name.mp4 \
    --wav $wav_file
