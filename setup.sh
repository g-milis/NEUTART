#!/bin/bash
set -e


urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


# Create non-synced directories
mkdir -p checkpoints
mkdir -p output
mkdir -p processed_data
mkdir -p processed_videos
mkdir -p tts_data


echo "Downloading pretrained models and assets..."

if [ ! -f spectre/data/generic_model.pkl ]; then
    echo -e "\nPlease enter your FLAME credentials. If you do not have an account, register at https://flame.is.tue.mpg.de/"
    read -p "Username:" username
    read -p "Password:" password
    username=$(urle $username)
    password=$(urle $password)

    echo -e "\nDownloading FLAME 3D model..."
    wget --post-data "username=$username&password=$password" \
        "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1" \
        --no-check-certificate --continue  -O FLAME2020.zip
    unzip FLAME2020.zip -d spectre/data/FLAME2020
    rm FLAME2020.zip
    mv spectre/data/FLAME2020/generic_model.pkl spectre/data
    rm -r spectre/data/FLAME2020
fi

if [ ! -f spectre/pretrained/spectre_model.tar ]; then
    echo -e "\nDownloading pretrained SPECTRE model for 3D facial reconstruction..."
    mkdir -p spectre/pretrained
    gdown --no-cookies https://drive.google.com/uc?id=1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B -O spectre/pretrained/spectre_model.tar
fi

if [ ! -d assets/55F ]; then
    echo -e "\nDownloading inference assets for pretrained models..."
    mkdir -p processed_data/21M
    mkdir -p processed_data/55F
    gdown --no-cookies https://drive.google.com/uc?id=17hU4KQ6Yh1byrVqWcZXEtM5yPvR2DABm
    unzip assets.zip
    rm assets.zip
    if [ ! -d processed_data/21M/stats.json ]; then
        mv assets/21M/stats.json processed_data/21M
    fi
    if [ ! -d processed_data/55F/stats.json ]; then
        mv assets/55F/stats.json processed_data/55F
    fi
fi

if [ ! -f checkpoints/21M/photorealistic/latest_net_G.pth ]; then
    echo -e "\nDownloading pretrained models for subject 21M..."
    gdown --no-cookies https://drive.google.com/uc?id=1Q3es8h1D34cpebdDPsauzHm3DE8k1Z4X
    unzip 21M.zip -d checkpoints
    rm 21M.zip
fi

if [ ! -f checkpoints/55F/photorealistic/latest_net_G.pth ]; then
    echo -e "\nDownloading pretrained models for subject 55F..."
    gdown --no-cookies https://drive.google.com/uc?id=1uYc1pxcG4VhrFy4qbX8wqvlHvOiaSx9T
    unzip 55F.zip -d checkpoints
    rm 55F.zip
fi

if [ ! -f audiovisual/hifigan/generator_universal.pth.tar ]; then
    echo -e "\nDownloading HiFI-GAN vocoder..."
    wget https://github.com/ming024/FastSpeech2/raw/master/hifigan/generator_universal.pth.tar.zip
    unzip generator_universal.pth.tar.zip -d audiovisual/hifigan
    rm generator_universal.pth.tar.zip
fi

if [ ! -f checkpoints/TCD-TIMIT/TCD-TIMIT.pth.tar ]; then
    echo -e "\nDownloading pretrained multispeaker model on TCD-TIMIT for training initialization..."
    mkdir -p checkpoints/TCD-TIMIT
    gdown --no-cookies https://drive.google.com/uc?id=1lPsqzycumpBd84bzOziJ-bdH6sYVHb6w -O checkpoints/TCD-TIMIT/TCD-TIMIT.pth.tar
fi

if [ ! -f checkpoints/meta-renderer/15_net_G.pth ]; then
    echo -e "\nDownloading pretrained meta-renderer for training initialization..."
    mkdir -p checkpoints/meta-renderer
    gdown --no-cookies https://drive.google.com/uc?id=1rvLxlE6WL6B0XKJJ93UoWhGc-i9stfcL
    unzip checkpoints_meta-renderer.zip -d checkpoints
    rm checkpoints_meta-renderer.zip
    mv checkpoints/checkpoints_meta-renderer/* checkpoints/meta-renderer
    rmdir checkpoints/checkpoints_meta-renderer
fi

if [ ! -f audiovisual/visual_sr/data/LRS3_V_WER32.3/model.pth ]; then
    echo -e "\nDownloading lip-reading model for perceptual loss..."
    gdown --no-cookies https://drive.google.com/uc?id=1yHd4QwC7K_9Ro2OM_hC7pKUT2URPvm_f
    unzip LRS3_V_WER32.3.zip -d audiovisual/visual_sr/data
    rm LRS3_V_WER32.3.zip
fi

echo "Setup complete."
