#!/bin/bash
set -e


# Input text
text="This is a demonstration for our code."
# Base name of the output files (.wav, .mp4, etc...)
name=demo
# Reference subject (dataset)
subject=21M


# Synthesize the audio and the 3D talking head from the input text
python audiovisual/synthesize.py \
    --text "$text" \
    --dataset $subject \
    --name $name

# Use the 3D talking head to render a photorealistic video
bash photorealistic/render.sh $subject $name
