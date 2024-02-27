import argparse
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch
from skimage.transform import warp
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from spectre.config import cfg as spectre_cfg
from spectre.datasets import datasets
from spectre.src.spectre import SPECTRE
from spectre.src.utils import util
import photorealistic.utils as utils


def pad_5(items):
    items.insert(0, items[0])
    items.insert(0, items[0])
    items.append(items[-1])
    items.append(items[-1])
    return items


def main():
    print("3D face reconstruction with SPECTRE...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_dir", type=str, help="Path to subject folder.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Whether the videos are intended for training the renderer or running inference.")
    args = parser.parse_args()

    device = "cuda"

    # Load images
    images_folder = os.path.join(args.subject_dir, "images")
    dataset = datasets.TestData(images_folder, device=device)

    spectre = SPECTRE(spectre_cfg, device)

    data = [datum for datum in tqdm(dataset)]
    image_paths = [datum["imagepath"] for datum in data]
    images = [datum["image"] for datum in data]
    tforms = [datum["tform"] for datum in data]
    original_sizes = [datum["original_size"] for datum in data]

    # Pad
    image_paths = pad_5(image_paths)
    tforms = pad_5(tforms)
    images = pad_5(images)
    original_sizes = pad_5(original_sizes)

    # Chunk size
    L = 55

    # Create lists of overlapping indices
    indices = list(range(len(image_paths)))
    overlapping_indices = [indices[i:(i + L)] for i in range(0, len(indices), L - 4)]

    if len(overlapping_indices[-1]) < 5:
        # If the last chunk has less than 5 frames, pad it with the semilast frame
        overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
        overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
        overlapping_indices = overlapping_indices[:-1]

    overlapping_indices = np.array(overlapping_indices)

    # Do this to index with multiple indices
    image_paths = np.array(image_paths)
    images = np.array(images, dtype=object)
    tforms = np.array(tforms, dtype=object)
    original_sizes = np.array(original_sizes)

    all_codedicts = []
    if args.mode == "train":
        all_shape_images = []
        all_nmfc_images = []

    with torch.no_grad():
        for chunk_id in range(len(overlapping_indices)):
            images_chunk = images[overlapping_indices[chunk_id]]
            tforms_chunk = tforms[overlapping_indices[chunk_id]]
            original_sizes_chunk = original_sizes[overlapping_indices[chunk_id]]

            images_array = np.stack([img for img in images_chunk])
            images_array = torch.from_numpy(images_array).type(dtype=torch.float32).to(device)

            codedict, initial_deca_exp, initial_deca_jaw = spectre.encode(images_array)
            codedict["exp"] = codedict["exp"] + initial_deca_exp
            codedict["pose"][..., 3:] = codedict["pose"][..., 3:] + initial_deca_jaw

            for key in codedict:
                if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                    pass
                elif chunk_id == 0:
                    codedict[key] = codedict[key][:-2]
                elif chunk_id == len(overlapping_indices) - 1:
                    codedict[key] = codedict[key][2:]
                else:
                    codedict[key] = codedict[key][2:-2]

            if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                pass
            elif chunk_id == 0:
                tforms_list = tforms_chunk[:-2]
                original_sizes_list = original_sizes_chunk[:-2]
            elif chunk_id == len(overlapping_indices) - 1:
                tforms_list = tforms_chunk[2:]
                original_sizes_list = original_sizes_chunk[2:]
            else:
                tforms_list = tforms_chunk[2:-2]
                original_sizes_list = original_sizes_chunk[2:-2]

            new_codedict = {}
            for key in codedict:
                new_codedict[key] = codedict[key].cpu().numpy()
            new_codedict["tform"] = tforms_list
            new_codedict["original_size"] = original_sizes_list

            # Convert the dictionary of lists to a list of dictionaries
            for i in range(len(new_codedict["images"])):
                single_codedict = {}
                for k, v in new_codedict.items():
                    if k in ["tform", "original_size"]:
                        new_v = v[i]
                    else:
                        new_v = v[i][None, ...]
                    single_codedict[k] = new_v
                all_codedicts.append(single_codedict)

            _, visdict = spectre.decode(codedict)

            if args.mode == "train":
                all_shape_images += [image for image in visdict["shape_images"]]
                all_nmfc_images += [image for image in visdict["nmfc_images"]]

    # Remove padding
    image_paths = image_paths[2:-2]
    all_codedicts = all_codedicts[2:-2]
    if args.mode == "train":
        all_shape_images = all_shape_images[2:-2]
        all_nmfc_images = all_nmfc_images[2:-2]

    assert len(all_codedicts) == len(image_paths)

    print("Saving reconstructions...")
    for i in tqdm(range(len(all_codedicts))):
        img_pth = image_paths[i]

        utils.mkdir(os.path.dirname(img_pth.replace("/images", "/SPECTRE")))
        codedict_pth = os.path.splitext(img_pth.replace("/images", "/SPECTRE"))[0] + ".pkl"
        with open(codedict_pth, "wb") as f:
            pickle.dump(all_codedicts[i], f)

        if args.mode == "train":
            utils.mkdir(os.path.dirname(img_pth.replace("/images", "/shapes")))
            shape_pth = img_pth.replace("/images", "/shapes").replace(".png", ".jpg")
            shape_image = warp(
                util.tensor2image(all_shape_images[i]) / 255,
                dataset[i]["tform"],
                output_shape=(
                    dataset[i]["original_size"][1],
                    dataset[i]["original_size"][0]
                )
            )
            cv2.imwrite(shape_pth, (shape_image * 255).astype(int))

            utils.mkdir(os.path.dirname(img_pth.replace("/images", "/nmfcs")))
            nmfc_pth = img_pth.replace("/images", "/nmfcs").replace(".png", ".jpg")
            nmfc_image = warp(
                util.tensor2image(all_nmfc_images[i]) / 255,
                dataset[i]["tform"],
                output_shape=(
                    dataset[i]["original_size"][1],
                    dataset[i]["original_size"][0]
                )
            )
            cv2.imwrite(nmfc_pth, (nmfc_image * 255).astype(int))


if __name__=="__main__":
    main()
