import json

import librosa
from scipy.io import wavfile
import skvideo.io
import torchvision


subject = "HDTF_25"

input_wav = f"assets/{subject}.wav"
input_mp4 = f"assets/{subject}.mp4"
segments_json = f"assets/segments.json"
output_path = "assets/HDTF"


# Load JSON data
with open(segments_json) as f:
    segments = json.load(f)["segments"]

# Load audio file
audio, sr = librosa.load(input_wav, sr=22050)

video = skvideo.io.vread(input_mp4)
fps = 25

# Process each segment
for segment in segments:
    # Extract segment
    segment_audio = audio[int(segment["start"] * sr):int(segment["end"] * sr)]
    segment_video = video[int(segment["start"] * fps):int(segment["end"] * fps)]

    # Save audio segment
    output_wav = f"{subject}/tts/{segment['id']:03d}.wav"
    wavfile.write(output_wav, sr, segment_audio)

    # Save video segment
    torchvision.io.write_video(
        f"{subject}/videos/{segment['id']:03d}.mp4", segment_video, fps=fps
    )

    # Save text to a corresponding text file
    output_txt = f"{subject}/tts/{segment['id']:03d}.lab"
    with open(output_txt, "w") as txt_file:
        txt_file.write(segment["text"])
