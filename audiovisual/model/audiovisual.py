import json
import os

import torch.nn as nn

from transformer import (
    Encoder,
    AudioDecoder,
    VisualDecoder
)
from utils.tools import get_mask_from_lengths

from .modules import VarianceAdaptor


class Audiovisual(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Configs
        (preprocess_config, model_config, train_config) = configs
        self.model_config = model_config
        self.audiovisual = "b" in train_config["losses"]

        # Modules
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = AudioDecoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )
        self.blendshape_decoder = VisualDecoder(model_config)
        self.blendshape_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["FLAME_channels"]
        )
        # Use speaker embeddings for multi-speaker model
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(preprocess_config["path"]["processed_path"], "speakers.json")
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        batch,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0
    ):
        speakers = batch["speakers"]
        texts = batch["phones"]
        src_lens = batch["src_lens"]
        max_src_len = batch["max_src_len"]

        if len(batch.keys()) > 8:
            mel_lens = batch["mel_lens"]
            max_mel_len = batch["max_mel_len"]
            p_targets = batch["pitches"]
            e_targets = batch["energies"]
            d_targets = batch["durations"]
        else:
            mel_lens = None
            max_mel_len = None
            p_targets = None
            e_targets = None
            d_targets = None

        # Text (phoneme) and mel masks
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        # Encode phoneme sequence 
        # texts.shape: [BATCH, PHON_SEQ_LEN]
        # output.shape: [BATCH, PHON_SEQ_LEN, TRANS_HID_SIZE (256)]
        output = self.encoder(texts, src_masks)
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Forward through variance adaptor
        # new output.shape: [BATCH, MEL_FRAMES, TRANS_HID_SIZE (256)]
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control
        )

        # Forward intermediate representation through FLAME decoder
        blendshape_predictions, _ = self.blendshape_decoder(output, mel_masks)
        blendshape_predictions = self.blendshape_linear(blendshape_predictions)
    
        # Forward intermediate representation through audio decoder to get spectrogram
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        return {
            "mels": output,
            "pitches": p_predictions,
            "energies": e_predictions,
            "log_durations": log_d_predictions,
            "durations_rounded": d_rounded,
            "blendshapes": blendshape_predictions,
            "src_masks": src_masks,
            "mel_masks": mel_masks,
            "src_lens": src_lens,
            "mel_lens": mel_lens
        }
