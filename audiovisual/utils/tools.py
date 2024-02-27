import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from moviepy.editor import AudioFileClip, VideoFileClip
from scipy.io import wavfile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if len(data.keys()) > 10:
        ids = data["ids"]
        raw_texts = data["raw_texts"]
        speakers = data["speakers"]
        phones = data["phones"]
        src_lens = data["src_lens"]
        max_src_len = data["max_src_len"]
        mels = data["mels"]
        mel_lens = data["mel_lens"]
        max_mel_len = data["max_mel_len"]
        pitches = data["pitches"]
        energies = data["energies"]
        durations = data["durations"]
        try:
            blendshapes = data["blendshapes"]
            blendshapes = torch.from_numpy(blendshapes).float().to(device, non_blocking=True)
        except KeyError:
            blendshapes = None

        speakers = torch.from_numpy(speakers).long().to(device, non_blocking=True)
        phones = torch.from_numpy(phones).long().to(device, non_blocking=True)
        src_lens = torch.from_numpy(src_lens).to(device, non_blocking=True)
        mels = torch.from_numpy(mels).float().to(device, non_blocking=True)
        mel_lens = torch.from_numpy(mel_lens).to(device, non_blocking=True)
        pitches = torch.from_numpy(pitches).float().to(device, non_blocking=True)
        energies = torch.from_numpy(energies).to(device, non_blocking=True)
        durations = torch.from_numpy(durations).long().to(device, non_blocking=True)

        try:
            mouths = data["mouths"]
            cams = data["cams"]
        except KeyError:
            mouths = None
            cams = None

        return {
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "phones": phones,
            "src_lens": src_lens,
            "max_src_len": max_src_len,
            "mels": mels,
            "mel_lens": mel_lens,
            "max_mel_len": max_mel_len,
            "pitches": pitches,
            "energies": energies,
            "durations": durations,
            "blendshapes": blendshapes,
            "mouths": mouths,
            "cams": cams
        }

    if len(data.keys()) <= 10:
        ids = data["ids"]
        raw_texts = data["raw_texts"]
        speakers = data["speakers"]
        phones = data["phones"]
        src_lens = data["src_lens"]
        max_src_len = data["max_src_len"]
        
        speakers = torch.from_numpy(speakers).long().to(device)
        phones = torch.from_numpy(phones).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return {
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "phones": phones,
            "src_lens": src_lens,
            "max_src_len": max_src_len
        }


def log(
    logger, step=None, losses=None, fig=None, audio=None, video=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        for tag, loss in losses.items():
            logger.add_scalar(f"Loss/{tag}", loss, step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )
    
    if video is not None:
        # (N, T, C, H, W)
        logger.add_video(
            tag,
            video,
            fps=86
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_samples(predictions, vocoder, model_config, preprocess_config):
    from .model import vocoder_infer
    
    mel_predictions = predictions["mels"].transpose(1, 2)
    lengths = predictions["mel_lens"] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )
    return wav_predictions


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for _, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def write_video_with_sound(outfile, video, audio, fps=22050/256, sr=22050):
    basename = outfile.replace(".mp4", "")
    # Write to a temporary mp4 without sound
    videopath = f"{basename}_tmp.mp4"
    torchvision.io.write_video(videopath, video, fps=fps)
    video = VideoFileClip(videopath)
    os.remove(videopath)
    # Write a temporary wav and load it
    wavpath = f"{basename}_tmp.wav"
    wavfile.write(wavpath, sr, audio.astype(np.int16))
    video_audio = AudioFileClip(wavpath)
    os.remove(wavpath)
    # Join video with audio and write back
    video = video.set_audio(video_audio)
    video.write_videofile(outfile, audio_codec="mp3", logger=None)
