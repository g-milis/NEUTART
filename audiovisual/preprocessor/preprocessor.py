import json
import os
import pickle
import random

from glob import glob
import librosa
import numpy as np
import pyworld as pw
import skimage.io
import skvideo.io
import tgt
import torch
import torchvision
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
import utils.lipread as lipread_utils
from spectre.utils.data_utils import landmarks_interpolate


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["tts_path"]
        self.videos_path = config["path"]["videos_path"]
        self.processed_videos_path = config["path"]["processed_videos_path"]
        self.out_dir = config["path"]["processed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.fps = config["fps"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )


    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "blendshapes")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mouth")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "cam")), exist_ok=True)

        print("Processing data...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        index = 0
        for speaker in os.listdir(self.in_dir):
            speakers[speaker] = index
            index += 1
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, speaker, f"{basename}.TextGrid"
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)
                else:
                    continue

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        # Perform normalization
        print("Computing statistic quantities...")
        pitch_mean = pitch_scaler.mean_[0]
        pitch_std = pitch_scaler.scale_[0]
        energy_mean = energy_scaler.mean_[0]
        energy_std = energy_scaler.scale_[0]

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std)
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std)
                ]
            }

            f.write(json.dumps(stats))

        print(
            f"""Total time: {
                n_frames * self.hop_length / self.sampling_rate / 3600
            } hours"""
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[:self.val_size]:
                f.write(m + "\n")

        return out


    def process_utterance(self, speaker, basename):
        # Textgrid path (TTS alignment)
        tg_path = os.path.join(
            self.out_dir, speaker, f"{basename}.TextGrid"
        )
        # Audio path
        wav_path = os.path.join(self.in_dir, speaker, f"{basename}.wav")
        # Transcription path
        text_path = os.path.join(self.in_dir, speaker, f"{basename}.lab")

        # Find video's index (from photorealistic preprocessing)
        try:
            with open(
                os.path.join(
                    self.processed_videos_path,
                    "boxes",
                    f"{basename}.txt"
                )
            ) as f:
                video_index = int(f.readline().split(" ")[-1])
        except FileNotFoundError:
            return None

        # Use that index to get the paths for 3DMMs, landmarks and frames
        spectre_paths = sorted(
            glob(
                os.path.join(
                    self.processed_videos_path,
                    "SPECTRE",
                    f"{video_index:03d}_*.pkl"
                )
            )
        )
        landmarks_paths = sorted(
            glob(
                os.path.join(
                    self.processed_videos_path,
                    "landmarks",
                    f"{video_index:03d}_*.txt"
                )
            )
        )
        frame_paths = sorted(
            glob(
                os.path.join(
                    self.processed_videos_path,
                    "images",
                    f"{video_index:03d}_*.png"
                )
            )
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start):int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path) as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[:sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Perform linear interpolation
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, :sum(duration)]
        energy = energy[:sum(duration)]

        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                pitch[i] = np.mean(pitch[pos:(pos + d)])
            else:
                pitch[i] = 0
            pos += d
        pitch = pitch[:len(duration)]

        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                energy[i] = np.mean(energy[pos:(pos + d)])
            else:
                energy[i] = 0
            pos += d
        energy = energy[:len(duration)]

        # Guard NaN values
        if np.isnan(pitch).any() or np.isnan(energy).any():
            return None

        # Get 3DMM parameters
        pose = []
        exp =[]
        cam = []
        for p in spectre_paths:
            with open(p, "rb") as f:
                codedict = pickle.load(f)
            pose.append(codedict["pose"])
            exp.append(codedict["exp"])
            cam.append(codedict["cam"][0])
        # Construct the blendshape vector of 56 elements
        try:
            blendshapes = np.array(
                np.hstack(
                    (np.vstack(pose), np.vstack(exp))
                )
            )
            cam = np.array(cam)
        except:
            return None

        # Get landmarks
        landmarks = []
        for f in landmarks_paths:
            landmarks.append(np.loadtxt(f)[14:, :])
        landmarks = np.array(landmarks_interpolate(landmarks))
        
        # Read video
        # video = skvideo.io.vread(video_path)
        frames = []
        for frame in frame_paths:
            frames.append(skimage.io.imread(frame))
        video = np.array(frames)
            

        # Trim
        blendshapes = blendshapes[
            int(self.fps * start):int(self.fps * end)
        ].astype(np.float32)
        video = video[
            int(self.fps * start):int(self.fps * end)
        ].astype(np.float32)/255.0
        landmarks = landmarks[
            int(self.fps * start):int(self.fps * end)
        ].astype(np.float32)
        cam = cam[
            int(self.fps * start):int(self.fps * end)
        ].astype(np.float32)

        # Cut and save the mouth for memory saving
        video = torch.tensor(video)
        landmarks = torch.tensor(landmarks)
        mouth = lipread_utils.cut_mouth(video, landmarks)

        # import cv2
        # img = 255 * video[0].numpy()
        # print(img.shape)
        # for landmark in landmarks[0]:
        #     x, y = landmark
        #     cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        # cv2.imwrite("test.jpg", img)

        # torchvision.io.write_video(
        #     os.path.join("mouth.mp4"),
        #     255 * mouth.unsqueeze(-1).repeat(1, 1, 1, 3),
        #     fps=25
        # )
        # raise

        # Interpolate blendshapes to the number of audio frames
        blendshapes = torch.nn.functional.interpolate(
            torch.tensor(
                blendshapes.T
            ).unsqueeze(0),
            size=sum(duration),
            mode="linear",
            align_corners=True
        ).squeeze().transpose(0, 1)

        # Save files
        dur_filename = f"{speaker}-duration-{basename}.npy"
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = f"{speaker}-pitch-{basename}.npy"
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = f"{speaker}-energy-{basename}.npy"
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = f"{speaker}-mel-{basename}.npy"
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        blendshapes_filename = f"{speaker}-blendshapes-{basename}.npy"
        np.save(os.path.join(self.out_dir, "blendshapes", blendshapes_filename), blendshapes)

        mouth_filename = f"{speaker}-mouth-{basename}.mp4"
        torchvision.io.write_video(
            os.path.join(self.out_dir, "mouth", mouth_filename),
            255 * mouth.unsqueeze(-1).repeat(1, 1, 1, 3),
            fps=25
        )

        cam_filename = f"{speaker}-cam-{basename}.npy"
        np.save(os.path.join(self.out_dir, "cam", cam_filename), cam)

        pitch = self.remove_outlier(pitch)
        energy = self.remove_outlier(energy)

        return (
            "|".join([basename, speaker, text, raw_text]),
            pitch,
            energy,
            mel_spectrogram.shape[1]
        )


    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time


    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]


    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
