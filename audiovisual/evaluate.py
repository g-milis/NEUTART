import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.dataset import Dataset
from audiovisual.model import AudiovisualLoss
from audiovisual.utils.render import render_blendshapes
from audiovisual.utils.tools import log, synth_samples, to_device, write_video_with_sound


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize(model, configs, vocoder, batchs):
    preprocess_config, model_config, train_config = configs

    for batch in batchs:
        with torch.no_grad():
            output = model(batch)
            wav = synth_samples(
                output,
                vocoder,
                model_config,
                preprocess_config
            )[0]
            length = output["mel_lens"][0]

            if 'b' in train_config['losses']:
                blendshapes = output["blendshapes"][0, :length, :].unsqueeze(0)

                print(blendshapes.shape)
                images, _ = render_blendshapes(blendshapes)
            else: 
                images = None
            return images, wav


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, _, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    # Get loss function
    Loss = AudiovisualLoss(train_config).to(device)
    N = len(dataset)

    max_batch = 20
    if N > max_batch:
        N = batch_size * max_batch

    # Evaluation
    n_losses = 10
    loss_sums = [0 for _ in range(n_losses)]
    for b, batchs in enumerate(loader):
        if b == max_batch: break
        for idx, batch in enumerate(batchs):
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(batch)

                # Calculate Loss
                losses = Loss(batch, output)

                for i, loss in enumerate(losses.values()):
                    loss_sums[i] += loss.item() * len(batch["ids"])

                if b == 0 and idx == 0 and step % train_config["step"]["val_step"] == 0:
                    model_name = train_config["path"]["new_checkpoint"].split(
                        os.sep
                    )[-1].split('.')[0].split('.')[-1]
                    os.makedirs(f"eval/{preprocess_config['dataset']}/{model_name}", exist_ok=True)
                    images, wav = synthesize(model, configs, vocoder, [batch])
                    if images is None:
                        from scipy.io import wavfile
                        wavfile.write(f"eval/{preprocess_config['dataset']}/{model_name}/{step}.wav", 22050, wav)
                    else:
                        write_video_with_sound(
                            f"eval/{preprocess_config['dataset']}/{model_name}/{step}.mp4",
                            images,
                            wav
                        )


    loss_means = [loss_sum / N for loss_sum in loss_sums]

    if logger is not None:
        losses = {tag: loss_means[i] for i, tag in enumerate(losses.keys())}
        log(logger, step, losses=losses)
