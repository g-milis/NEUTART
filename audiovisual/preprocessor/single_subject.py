import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from glob import glob
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.text import _clean_text


def prepare_align(config):
    video_dir = config["path"]["videos_path"]
    text_dir = config["path"]["text_path"]
    tts_dir = config["path"]["tts_path"]
    sr = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    speaker = config["dataset"]
    for video_file in tqdm(glob(video_dir)):
        if not video_file.endswith("mp4"):
            continue

        # Isolate uterance ID
        utterance = os.path.basename(video_file).replace(".mp4", "")
        # In my case, the video names contained the speaker, too
        utterance = utterance.replace(f"{config['dataset']}_", "")
        try:
            # Get text
            text_path = os.path.join(text_dir, f"{utterance.upper()}.txt")
            with open(text_path) as f:
                text = f.read().strip()
                # text = " ".join([x.split()[2].strip() for x in f.readlines()])
            text = _clean_text(text)

            # Read audio
            wav, _ = librosa.load(video_file, sr=sr)
            wav = wav / max(abs(wav)) * max_wav_value

        except:
            print("Skipping:", speaker, utterance)
            continue

        # Store in tts_dir
        wav_save_path = os.path.join(tts_dir, speaker, f"{utterance}.wav")
        wavfile.write(wav_save_path, sr, wav.astype(np.int16))

        text_save_path = os.path.join(tts_dir, speaker, f"{utterance}.lab")
        with open(text_save_path, "w") as f: 
            f.write(text)
