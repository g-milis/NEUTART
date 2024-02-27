import json
import os
import sys

import torch

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import hifigan
from model import Audiovisual, ScheduledOptim


def get_model(args, configs, device, train=False, transfer=False):
    (_, model_config, train_config) = configs
    model = Audiovisual(configs).to(device)

    if args.restore_step:
        if train:
            print("Loading previous audiovisual module...")
            ckpt_path = train_config["path"]["prev_checkpoint"]
        else:
            print("Loading audiovisual module...")
            ckpt_path = train_config["path"]["new_checkpoint"]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"], strict=False)

        # Freeze the text encoder
        if train and transfer:
            print("Only optimizing submodules:")
            for name, module in model.named_children():
                if name == "encoder":
                    module.requires_grad_(False)
                else: 
                    print(name)
            print()

    if not train:
        # Evaluation mode
        model.eval()
        model.requires_grad_ = False
        return model
    else:
        model.train()
        # Load optimizers for training
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        return model, scheduled_optim


def get_param_num(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def get_vocoder(config, device):
    with open("audiovisual/hifigan/config.json") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("audiovisual/hifigan/generator_universal.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder.to(device)


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1).cpu().numpy()

    wavs = (
        wavs * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
