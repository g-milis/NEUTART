import json
import os
import sys

import numpy as np
import skvideo.io
import torch
from torch.utils import data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.text import text_to_sequence
from audiovisual.utils.tools import pad_1D, pad_2D


class Dataset(data.Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.processed_path = preprocess_config["path"]["processed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.fps = preprocess_config["fps"]

        losses = train_config["losses"]
        self.lipreading = "l" in losses
        self.audiovisual = "b" in losses

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.processed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last


    def __len__(self):
        return len(self.text)


    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx]))

        mel_path = os.path.join(
            self.processed_path,
            "mel",
            f"{speaker}-mel-{basename}.npy"
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.processed_path,
            "pitch",
            f"{speaker}-pitch-{basename}.npy"
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.processed_path,
            "energy",
            f"{speaker}-energy-{basename}.npy"
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.processed_path,
            "duration",
            f"{speaker}-duration-{basename}.npy"
        )
        duration = np.load(duration_path)
        
        if self.audiovisual:
            blendshapes_path = os.path.join(
                self.processed_path,
                "blendshapes",
                f"{speaker}-blendshapes-{basename}.npy"
            )
            blendshapes = np.load(blendshapes_path)

            if self.lipreading:
                try:
                    mouth_path = os.path.join(
                        self.processed_path,
                        "mouth",
                        f"{speaker}-mouth-{basename}.mp4"
                    )
                    # Grayscale written as 3 channels
                    mouth = skvideo.io.vread(mouth_path)[..., 0]
                    cam_path = os.path.join(
                        self.processed_path,
                        "cam",
                        f"{speaker}-cam-{basename}.npy"
                    )
                    cam = np.load(cam_path)
                except FileNotFoundError:
                    return None

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        if self.audiovisual:
            sample.update({
                "blendshapes": blendshapes,
            })
            if self.lipreading:
                sample["mouth"] = mouth
                sample["cam"] = cam
        return sample


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


    def process_meta(self, filename):
        with open(
            os.path.join(self.processed_path, filename), encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text


    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        phones = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        if self.audiovisual:
            blendshapes = [data[idx]["blendshapes"] for idx in idxs]
            if self.lipreading:
                mouths = [data[idx]["mouth"] for idx in idxs]
                cams = [data[idx]["cam"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in phones])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        phones = pad_1D(phones)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        if self.audiovisual:
            blendshapes = pad_2D(blendshapes)
        sample = {
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "phones": phones,
            "src_lens": text_lens,
            "max_src_len": np.max(text_lens),
            "mels": mels,
            "mel_lens": mel_lens,
            "max_mel_len": np.max(mel_lens),
            "pitches": pitches,
            "energies": energies,
            "durations": durations,
        }
        if self.audiovisual:
            sample.update({
                "blendshapes": blendshapes,
            })
            if self.lipreading:
                sample["mouths"] = mouths
                sample["cams"] = cams
        return sample


    def collate_fn(self, data):
        data = [x for x in data if x is not None]

        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["processed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)


    def __len__(self):
        return len(self.text)


    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx]))
        return (basename, speaker_id, phone, raw_text)


    def process_meta(self, filename):
        with open(filename, encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text


    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        texts = pad_1D(texts)
        return {
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "phones": texts,
            "src_lens": text_lens,
            "max_src_len": np.max(text_lens)
        }


if __name__ == "__main__":
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from audiovisual.utils.tools import to_device


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("config/21M/preprocess.yaml"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("config/21M/train.yaml"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 1,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(f"Training set with size {len(train_dataset)} is composed of {n_batch} batches.")
    
    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(f"Validation set with size {len(val_dataset)} is composed of {n_batch} batches.")
