import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.preprocessor.preprocessor import Preprocessor


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
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
