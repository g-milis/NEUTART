import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.preprocessor import tcdtimit, single_subject


def main(config):
    if "TCD-TIMIT" in config["dataset"]:
        tcdtimit.prepare_align(config)
    else:
        single_subject.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset name (should have 3 files in ./config/dataset_name/*.yaml)",
        default="21M"
    )
    args = parser.parse_args()

    config = yaml.load(
        open(os.path.join("audiovisual", "config", args.dataset, "preprocess.yaml")),
        Loader=yaml.FullLoader
    )
    main(config)
