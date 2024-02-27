import argparse
import os
import pickle
import re
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch
from skimage import img_as_ubyte
from skimage.transform import warp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from spectre.config import cfg as spectre_cfg
from spectre.src.spectre import SPECTRE
from spectre.src.utils import util
import photorealistic.utils as utils


def read_3D_params(path, device="cuda"):
    """ 
    Returns a list of dictionaries containing the 3D parameters:
    ["shape", "tex", "exp", "pose", "cam", "light", "images", "tform", "original_size"]
    """
    pkl_files = [os.path.join(path, pkl) for pkl in sorted(os.listdir(path))]
    params = []
    for p in pkl_files:
        with open(p, "rb") as f:
            param = pickle.load(f)
        for key in param.keys():
            if key not in ["tform", "original_size"]:
                param[key] = torch.from_numpy(param[key]).to(device)
        params.append(param)

    return params, pkl_files


def read_from_audiovisual(path):
    """ 
    Returns a list of dictionaries containing the parameters: ["exp", "pose"]
    """
    blendshapes = torch.load(path)
    params = []
    for blendshape in blendshapes[0]:
        params.append({
            "pose": blendshape[:6].unsqueeze(0),
            "exp": blendshape[6:].unsqueeze(0)
        })
    return params


def read_eye_landmarks(path):
    txt_files = [os.path.join(path, txt) for txt in sorted(os.listdir(path))]
    eye_landmarks_left = []
    eye_landmarks_right = []
    for f in txt_files:
        if os.path.exists(f):
            left = np.concatenate([np.loadtxt(f)[0:6], np.loadtxt(f)[12:13]], axis=0)
            right = np.concatenate([np.loadtxt(f)[6:12], np.loadtxt(f)[13:14]], axis=0)
            eye_landmarks_left.append(left)
            eye_landmarks_right.append(right)
    return [eye_landmarks_left, eye_landmarks_right]


def main():
    print("Creating modified NMFCs and landmarks...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default="0", help="Negative value to use CPU, or greater equal than zero for GPU id.")
    parser.add_argument("--subject_dir", type=str, help="Path to subject folder")
    parser.add_argument("--exp_name", type=str, help="Subfolder for specific experiment")
    parser.add_argument("--input", type=str, help="Input blendshapes file in pth format")
    parser.add_argument("--no_eye_gaze", action="store_true", help="If specified, do not use eye-landmarks")
    args = parser.parse_args()

    # Figure out the device
    device = "cuda"

    # Read parameters and use input to drive generation
    src_codedicts, paths = read_3D_params(os.path.join(args.subject_dir, args.exp_name, "SPECTRE"), device=device)
    trg_codedicts = read_from_audiovisual(args.input)

    # Read src eye landmarks.
    if not args.no_eye_gaze:
        src_eye_landmarks = read_eye_landmarks(os.path.join(args.subject_dir, "landmarks"))

    # Create save dirs
    utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "nmfcs"))
    utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "shapes"))
    utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "shapes_aligned"))
    utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "nmfcs_aligned"))
    if not args.no_eye_gaze:
        utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "landmarks"))
        utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "landmarks_aligned"))

    # Run SPECTRE decoding
    spectre = SPECTRE(config=spectre_cfg, device=device)

    # While the target sequence is longer then the reference,
    # keep adding copies of the reference in reverse and back
    while len(trg_codedicts) > len(src_codedicts):
        src_codedicts = src_codedicts + src_codedicts[::-1]
        paths = paths + paths[::-1]
        src_eye_landmarks = [
            src_eye_landmarks[0] + src_eye_landmarks[0][::-1],
            src_eye_landmarks[1] + src_eye_landmarks[1][::-1]
        ]

    for i, (src_codedict, trg_codedict, pth) in enumerate(zip(src_codedicts, trg_codedicts, paths)):  
        # Rename path to ensure consistency in case there are reversed copies
        new_pth = os.path.join(
            os.path.dirname(pth),
            f"{i:04d}-{os.path.basename(pth)}"
        )

        src_codedict["exp"] = trg_codedict["exp"]
        src_codedict["pose"][:, 3:6] = trg_codedict["pose"][:, 3:6]

        opdict, visdict = spectre.decode(src_codedict)

        nmfc_pth = os.path.splitext(new_pth.replace("/SPECTRE", "/nmfcs"))[0] + ".jpg"
        nmfc_image = warp(util.tensor2image(visdict["nmfc_images"][0])/255, src_codedict["tform"], output_shape=(src_codedict["original_size"][1], src_codedict["original_size"][0]))
        nmfc_image = img_as_ubyte(nmfc_image)
        cv2.imwrite(nmfc_pth, nmfc_image)

        mat_pth = os.path.splitext(pth.replace(f"/{args.exp_name}/SPECTRE", "/align_transforms"))[0] + ".txt"
        mat = np.loadtxt(mat_pth)

        nmfc_image_a = cv2.warpAffine(nmfc_image, mat, (nmfc_image.shape[1], nmfc_image.shape[0]), flags=cv2.INTER_LANCZOS4)
        cv2.imwrite(nmfc_pth.replace("/nmfcs", "/nmfcs_aligned"), nmfc_image_a)

        shape_pth = os.path.splitext(new_pth.replace("/SPECTRE", "/shapes"))[0] + ".jpg"
        shape_image = warp(util.tensor2image(visdict["shape_images"][0])/255, src_codedict["tform"], output_shape=(src_codedict["original_size"][1], src_codedict["original_size"][0]))
        shape_image = img_as_ubyte(shape_image)
        cv2.imwrite(shape_pth, shape_image)

        shape_image_a = cv2.warpAffine(shape_image, mat, (shape_image.shape[1], shape_image.shape[0]), flags=cv2.INTER_LANCZOS4)
        cv2.imwrite(shape_pth.replace("/shapes", "/shapes_aligned"), shape_image_a)

        # Adapt eye pupil and save eye landmarks
        if not args.no_eye_gaze:
            trg_lnds = src_codedict["tform"].inverse(112 + 112*opdict["landmarks2d"][0].cpu().numpy())
            trg_left_eye = trg_lnds[36:42]
            trg_right_eye = trg_lnds[42:48]

            src_left_eye = src_eye_landmarks[0][i]
            src_right_eye = src_eye_landmarks[1][i]

            src_left_center = np.mean(src_left_eye[0:6], axis=0, keepdims=True)
            src_right_center = np.mean(src_right_eye[0:6], axis=0, keepdims=True)

            trg_left_center = np.mean(trg_left_eye[0:6], axis=0, keepdims=True)
            trg_right_center = np.mean(trg_right_eye[0:6], axis=0, keepdims=True)

            trg_left_pupil = src_left_eye[6:7] + (trg_left_center - src_left_center)
            trg_right_pupil = src_right_eye[6:7] + (trg_right_center - src_right_center)

            eye_lnds = np.concatenate([trg_left_eye, trg_right_eye, trg_left_pupil, trg_right_pupil], axis=0).astype(np.int32)
            eye_lnds_pth = os.path.splitext(new_pth.replace("/SPECTRE", "/landmarks"))[0] + ".txt"
            np.savetxt(eye_lnds_pth, eye_lnds)

            eye_lnds_a = utils.transform_points(eye_lnds, mat)
            np.savetxt(eye_lnds_pth.replace("/landmarks", "/landmarks_aligned"), eye_lnds_a)


if __name__=="__main__":
    main()
