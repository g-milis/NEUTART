import argparse
import os
from os.path import join as pjoin
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.dataset import Dataset
from audiovisual.evaluate import evaluate
from audiovisual.model import AudiovisualLoss
from audiovisual.utils.model import get_model, get_param_num, get_vocoder
from audiovisual.utils.tools import log, to_device


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True, transfer=False)
    
    model = nn.DataParallel(model)
    Loss = AudiovisualLoss(train_config).to(device)
    print("Number of parameters:", get_param_num(model))

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    os.makedirs(os.path.dirname(train_config["path"]["new_checkpoint"]), exist_ok=True)
    for p in train_config["path"].values():
        if p.endswith("path"):
            os.makedirs(p, exist_ok=True)
    train_log_path = pjoin(train_config["path"]["log_path"], "train")
    val_log_path = pjoin(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                
                try:
                    _, losses = train_step(
                        step,
                        batch,
                        model,
                        optimizer,
                        Loss,
                        train_config
                    )
                except RuntimeError as err:
                    print(err)
                    continue

                if step % log_step == 0:
                    log(train_logger, step, losses=losses)

                if step % val_step == 0:
                    model.eval()
                    evaluate(model, step, configs, val_logger, vocoder)
                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        train_config["path"]["new_checkpoint"]
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


def train_step(step, batch, model, optimizer, Loss, train_config):
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    
    output = model(batch)
    # Calculate all losses
    losses = Loss(batch, output)
    total_loss = losses["total_loss"]

    # Backward
    total_loss = total_loss / grad_acc_step
    total_loss.backward()
    if step % grad_acc_step == 0:
        # Clipping gradients to avoid gradient explosion
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        # Update weights
        optimizer.step_and_update_lr()
        optimizer.zero_grad()
    return output, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=1)
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset name (should have 3 files in ./config/dataset_name/*.yaml)",
        default="21M"
    )
    args = parser.parse_args()

    # Read configs
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

    train_config["path"]["log_path"] = pjoin(
        "checkpoints",
        args.dataset,
        "audiovisual",
        "log",
        os.path.basename(train_config["path"]["new_checkpoint"]).replace(".pth.tar", "")
    )

    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
