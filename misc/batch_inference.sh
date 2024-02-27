#! /bin/bash
set -e


# Use an input file containing lines of text to be uttered
input=misc/utterances.txt
# Which subject to use
subject=21M


i=0
while IFS= read -r line
do
    # Input text
    text="$line"
    # Base name of the output files (.wav, .mp4, etc...)
    name=$i

    # Synthesize the audio and the 3D talking head from the input text
    python audiovisual/synthesize.py \
        --text "$text" \
        --dataset $subject \
        --name $name

    # Use the 3D talking head to render a photorealistic video
    bash photorealistic/render.sh $subject $name

    i=$(($i+1))
    
done < $input
