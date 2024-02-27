# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os

import torch

# * -------------------- general -------------------- *
def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json).

    Args:
        model_path (str): Model path.
        conf_path (str): Optional model config path.

    Returns:
        list[int, int, dict[str, Any]]: Config information loaded from json file.

    """
    if conf_path is None:
        model_conf = os.path.dirname(model_path) + "/model.json"
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info("reading a config file from " + model_conf)
        confs = json.load(f)
    if isinstance(confs, dict):
        # for lm
        args = confs
        return argparse.Namespace(**args)
    else:
        # for asr, tts, mt
        idim, odim, args = confs
        return idim, odim, argparse.Namespace(**args)


def torch_load(path, model):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.

    """
    if "snapshot" in os.path.basename(path):
        model_state_dict = torch.load(path, map_location=lambda storage, _: storage)[
            "model"
        ]
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, _: storage)

    if hasattr(model, "module"):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict
