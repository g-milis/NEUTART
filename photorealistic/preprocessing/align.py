import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import photorealistic.utils as utils


def save_aligned(mat_pths, aligned, args):
    # Make dirs
    out_paths = []
    out_paths += [p.replace("/align_transforms/", "/masks_aligned/") for p in mat_pths]
    out_paths += [p.replace("/align_transforms/", "/landmarks_aligned/") for p in mat_pths]
    if args.mode == "train":
        out_paths += [p.replace("/align_transforms/", "/faces_aligned/") for p in mat_pths]
        out_paths += [p.replace("/align_transforms/", "/shapes_aligned/") for p in mat_pths]
        out_paths += [p.replace("/align_transforms/", "/nmfcs_aligned/") for p in mat_pths]

    out_paths = set(os.path.dirname(out_pth) for out_pth in out_paths)
    for out_path in out_paths:
        utils.mkdir(out_path)

    for al, mat_pth in zip(aligned, mat_pths):
        al_iter = iter(al)

        mask_a = next(al_iter)
        mask_file = os.path.splitext(mat_pth.replace("/align_transforms/", "/masks_aligned/"))[0] + ".jpg"
        cv2.imwrite(mask_file, mask_a)

        lands_a = next(al_iter)
        lands_file = mat_pth.replace("/align_transforms/", "/landmarks_aligned/")
        np.savetxt(lands_file, lands_a)

        if args.mode == "train":
            face_a = next(al_iter)
            face_file = os.path.splitext(mat_pth.replace("/align_transforms/", "/faces_aligned/"))[0] + ".png"
            cv2.imwrite(face_file, face_a)

            shape_a = next(al_iter)
            shape_file = os.path.splitext(mat_pth.replace("/align_transforms/", "/shapes_aligned/"))[0] + ".jpg"
            cv2.imwrite(shape_file, shape_a)

            nmfc_a = next(al_iter)
            nmfc_file = os.path.splitext(mat_pth.replace("/align_transforms/", "/nmfcs_aligned/"))[0] + ".jpg"
            cv2.imwrite(nmfc_file, nmfc_a)


def align(mat_paths, args):
    rets = []
    for mat_pth in tqdm(mat_paths):
        ret = []
        mat = np.loadtxt(mat_pth)

        mask_pth = os.path.splitext(mat_pth.replace("/align_transforms/", "/masks/"))[0]+".jpg"
        mask = cv2.imread(mask_pth)
        mask_a = cv2.warpAffine(mask, mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
        ret.append( mask_a)

        land_file = mat_pth.replace("/align_transforms", "/landmarks/")
        lands = np.loadtxt(land_file)
        lands_a = utils.transform_points(lands, mat)
        ret.append(lands_a)

        if args.mode == "train":
            face_pth = os.path.splitext(mat_pth.replace("/align_transforms/", "/faces/"))[0]+".png"
            face = cv2.imread(face_pth)
            face_a = cv2.warpAffine(face, mat, (face.shape[1], face.shape[0]), flags=cv2.INTER_LANCZOS4)
            face_a[np.where(mask_a == 0)] = 0
            ret.append(face_a)

            s_pth = os.path.splitext(mat_pth.replace("/align_transforms/", "/shapes/"))[0]+".jpg"
            shape = cv2.imread(s_pth)
            shape_a = cv2.warpAffine(shape, mat, (shape.shape[1], shape.shape[0]), flags=cv2.INTER_LANCZOS4)
            ret.append(shape_a)

            n_pth = os.path.splitext(mat_pth.replace("/align_transforms/", "/nmfcs/"))[0]+".jpg"
            nmfc = cv2.imread(n_pth)
            nmfc_a = cv2.warpAffine(nmfc, mat, (nmfc.shape[1], nmfc.shape[0]), flags=cv2.INTER_LANCZOS4)
            ret.append(nmfc_a)
        
        rets.append(ret)

    return rets


def main():
    print("Face alignment...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_dir", type=str, help="Path to subject folder.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Whether the videos are intended for training the renderer or running inference.")
    args = parser.parse_args()

    # Get the path of each transformation file.
    mats_dir = os.path.join(args.subject_dir, "align_transforms")
    mat_paths = utils.get_mats_paths(mats_dir)

    aligned = align(mat_paths, args)
    save_aligned(mat_paths, aligned, args)


if __name__=="__main__":
    main()
