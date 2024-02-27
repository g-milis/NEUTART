import os
from glob import glob

import jiwer
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


with open("utterances.txt") as f:
    transcriptions = [line.strip() for line in f.readlines()]


def asr_cer(audio_file):
    audio, _ = torchaudio.load(audio_file)
    input_features = tokenizer(audio.numpy().squeeze(), return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_features).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0].lower()
    idx = int(os.path.basename(file).split(".")[0].split('_')[-1]) - 1
    print(idx)

    reference = transcriptions[idx].lower()
    print("ref:", reference)
    print("hyp:", transcription)

    return jiwer.cer(reference, transcription)


if __name__ == "__main__":
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    regexes = [
        f"output/09F/*.wav"
    ]

    for regex in regexes:
        print(regex)

        cers = []
        for file in glob(regex):
            cer = asr_cer(file)
            print(cer)
            cers.append(cer)

        cer = np.mean(np.array(cers))
        print(f"CER: {cer * 100:.2f}%")
