import argparse
import re
import os
from os.path import join as pjoin
from string import punctuation
import sys

import numpy as np
import torch
import yaml
from g2p_en import G2p
from torch.utils.data import DataLoader
from scipy.io import wavfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.dataset import TextDataset
from audiovisual.text import text_to_sequence
from audiovisual.utils.model import get_model, get_vocoder
from audiovisual.utils.render import render_blendshapes
from audiovisual.utils.tools import synth_samples, to_device, write_video_with_sound


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon():
    lexicon = {}
    with open("audiovisual/text/librispeech-lexicon.txt") as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon()

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print(f"Raw Text Sequence: {text}")
    print(f"Phoneme Sequence: {phones}")
    sequence = np.array(text_to_sequence(phones))
    return np.array(sequence), phones


def synthesize(model, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(
                batch,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            wav = synth_samples(
                output,
                vocoder,
                model_config,
                preprocess_config
            )[0]
            wavfile.write(train_config["path"]["result_path"].replace("mp4", "wav"), 22050, wav)

            blendshapes = output["blendshapes"]
            blendshapes = torch.nn.functional.interpolate(
                blendshapes.transpose(-1, -2),
                scale_factor=25*(256/22050),
                mode="linear",
            ).transpose(-1, -2)
            torch.save(blendshapes, train_config["path"]["result_path"].replace("mp4", "pth"))

            images, _ = render_blendshapes(blendshapes, 256, full_pose=False)
            write_video_with_sound(
                train_config["path"]["result_path"].replace(".mp4", "_3D.mp4"),
                images,
                wav,
                fps=25
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=1)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="single",
        help="Synthesize a whole dataset or a single sentence."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a source file with format like train.txt and val.txt, for batch mode only.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This code will be cleaned soon.",
        help="Raw text to synthesize, for single-sentence mode only.",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker synthesis, for single-sentence mode only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="sample",
        help="Name of the output files.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset name (should have 3 files in ./config/dataset_name/*.yaml).",
        default="21M"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="Control the pitch of the utterance, larger value for higher pitch.",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="Control the energy of the utterance, larger value for larger volume.",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="Control the speed of the utterance, larger value for slower speaking rate.",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(pjoin("audiovisual", "config", args.dataset, "preprocess.yaml")),
        Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(pjoin("audiovisual", "config", args.dataset, "model.yaml")),
        Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(pjoin("audiovisual", "config", args.dataset, "train.yaml")),
        Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)

    # Output path
    os.makedirs(train_config["path"]["result_path"], exist_ok=True)
    train_config["path"]["result_path"] = pjoin(
        train_config["path"]["result_path"],
        f"{args.name}.mp4"
    )
    
    # Get model
    model = get_model(args, configs, device, train=False)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        texts = np.array([preprocess_english(args.text)[0]])

        text_lens = np.array([len(texts[0])])
        batchs = [{
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "phones": texts,
            "src_lens": text_lens,
            "max_src_len": max(text_lens),
        }]

    control_values = args.pitch_control, args.energy_control, args.duration_control
    synthesize(model, configs, vocoder, batchs, control_values)
