import sys
from configparser import ConfigParser

import torch
from torchvision.transforms import Resize
import torchvision.transforms.functional as F_v

sys.path.append("visual_sr")
from visual_sr.dataloader.transform import (
    CenterCrop,
    Compose,
    Identity,
    Normalize
)
from visual_sr.lipreading.model import Lipreading


def get_lip_reader():
    config = ConfigParser()
    config.read("audiovisual/config/lipread_config.ini")
    lip_reader = Lipreading(config, device="cuda")
    lip_reader.eval()
    lip_reader.model.eval()
    return lip_reader


def cut_mouth(images, landmarks):
    """ Adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""
    # Video resolutions:
    # TCD-TIMIT: 1920 x 1080
    # I use crop width and height of 300, as well as Resize((90, 90)) to center around the mouth.

    # You may need to adjust the following 3 paramaters =======================
    focus_size = 200
    _crop_width = 200
    _crop_height = 150
    # =========================================================================

    _window_margin = 12
    crop_size = (88, 88)
    mean, std = 0.421, 0.165
    mouth_transform = Compose([
        Resize((focus_size, focus_size)),
        Normalize(0.0, 1.0),
        CenterCrop(crop_size),
        Normalize(mean, std),
        Identity()
    ])

    mouth_sequence = []
    images = images.permute(0, 3, 1, 2)
    images = F_v.rgb_to_grayscale(images).squeeze()

    for frame_idx, frame in enumerate(images):
        window_margin = min(_window_margin//2, frame_idx, len(landmarks) - 1 - frame_idx)
        smoothed_landmarks = landmarks[(frame_idx - window_margin):(frame_idx + window_margin + 1)].mean(dim=0)
        smoothed_landmarks += landmarks[frame_idx].mean(dim=0) - smoothed_landmarks.mean(dim=0)

        center_x, center_y = torch.mean(smoothed_landmarks, dim=0)
        center_x = center_x.round()
        center_y = center_y.round()
        height = _crop_height//2
        width = _crop_width//2
        threshold = 5

        img = frame

        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception("Too much bias in height.")
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception("Too much bias in width.")

        if center_y + height > img.shape[-2]:
            center_y = img.shape[-2] - height
        if center_y + height > img.shape[-2] + threshold:
            raise Exception("Too much bias in height.")
        if center_x + width > img.shape[-1]:
            center_x = img.shape[-1] - width
        if center_x + width > img.shape[-1] + threshold:
            raise Exception("Too much bias in width.")

        mouth = img[
            ...,
            int(center_y - height):int(center_y + height),
            int(center_x - width):int(center_x + round(width))
        ]
        mouth_sequence.append(mouth)
        
    mouths = torch.stack(mouth_sequence, dim=0)
    return mouth_transform(mouths)
