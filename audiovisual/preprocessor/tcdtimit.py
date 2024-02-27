import os
import sys

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["tts_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    for subset in ["volunteers", "lipspeakers"]:
        for speaker in tqdm(os.listdir(os.path.join(in_dir, subset))):
            speaker_dir = os.path.join(in_dir, subset, speaker, "Clips", "straightcam")
            for file_name in os.listdir(speaker_dir):
                if file_name[-4:] != ".txt":
                    continue
                # Isolate uterance ID
                utterance = file_name[:-4].lower()
                try:
                    # Get text
                    text_path = os.path.join(speaker_dir, file_name)
                    with open(text_path) as f:
                        text = f.readlines()
                        if subset == "volunteers":
                            text = " ".join([x.split()[2].strip() for x in text])
                        else:
                            text = text[0].strip()
                    text = _clean_text(text)

                    # Read wav
                    wav_path = os.path.join(
                        "/path/to/wavs",
                        f"{speaker}_{utterance}.wav"
                    )
                    wav, _ = librosa.load(wav_path, sr=sampling_rate)
                    wav = librosa.to_mono(wav)
                    wav = wav / max(abs(wav)) * max_wav_value

                except:
                    print("Skipping:", speaker, utterance)
                    continue

                # Store in tts_data
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wavfile.write(
                    os.path.join(out_dir, speaker, f"{utterance}.wav"),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(os.path.join(out_dir, speaker, f"{utterance}.lab"), "w") as f: 
                    f.write(text)
