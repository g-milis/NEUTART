#!/bin/bash
set -e


mode=$1
subject_dir=$2
videos_path=$3


# Face detection
python photorealistic/preprocessing/detect_faces.py --videos_path $videos_path --subject_dir $subject_dir --mode $mode

# Facial landmark detection
python photorealistic/preprocessing/detect_landmarks.py --subject_dir $subject_dir

# Face segmentation (creation of masks)
python photorealistic/preprocessing/segment_faces.py --subject_dir $subject_dir --mode $mode

# Performs 3D reconstruction of the face
python photorealistic/preprocessing/reconstruct_faces.py --subject_dir $subject_dir --mode $mode

# Aligns faces to the center of the frame
python photorealistic/preprocessing/align.py --subject_dir $subject_dir --mode $mode
