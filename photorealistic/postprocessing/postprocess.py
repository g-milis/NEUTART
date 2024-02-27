import argparse
import os
import re
import sys

import cv2
import numpy as np
from skimage import img_as_float32, img_as_ubyte

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from image_blender import Blend
import photorealistic.utils as utils


def unalign(face_a_paths, args):
    faces = []
    print("Removing alignment from face images...")
    for face_a_pth in face_a_paths:
        mat_file = os.path.splitext(face_a_pth.replace(f"/{args.exp_name}/faces_aligned", "/align_transforms"))[0]+".txt"
        mask_file = face_a_pth.replace(f"/{args.exp_name}/faces_aligned", "/masks").replace(".png", ".jpg")
        face_a = cv2.imread(face_a_pth)

        mat_file = re.sub("/\d{4}-", "/", mat_file)
        mask_file = re.sub("/\d{4}-", "/", mask_file)
        mat = np.loadtxt(mat_file)
        mask = cv2.imread(mask_file)
        face = cv2.warpAffine(face_a, mat, (face_a.shape[1], face_a.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LANCZOS4)
        face[np.where(mask==0)] = 0

        faces.append(face)

    return faces


def load_bboxes(dir):
    # Returns list with bounding boxes
    boxes = []
    txt_files = [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if ".txt" in f]
    for t in txt_files:
        boxes.extend(list(np.loadtxt(t, skiprows=1).astype(int)))
    return boxes


def blend_and_save_images(face_pths, blender, args):
    utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "full_frames"))
    if args.save_images:
        utils.mkdir(os.path.join(args.subject_dir, args.exp_name, "images"))

    # Load original bounding boxes
    boxes_folder = os.path.join(args.subject_dir, "boxes")
    boxes = load_bboxes(boxes_folder)

    print("Blending faces with background...")
    for face_pth in face_pths:
        full_frame_pth = face_pth.replace(f"/{args.exp_name}/faces", "/full_frames").replace(".png", ".jpg")
        mask_pth = face_pth.replace(f"/{args.exp_name}/faces", "/masks").replace(".png", ".jpg")
        img_pth = face_pth.replace(f"/{args.exp_name}/faces", "/images")

        full_frame_pth = re.sub("/\d{4}-", "/", full_frame_pth)
        mask_pth = re.sub("/\d{4}-", "/", mask_pth)
        img_pth = re.sub("/\d{4}-", "/", img_pth)

        assert os.path.exists(full_frame_pth), f"Frame {full_frame_pth} does not exist."
        assert os.path.exists(mask_pth), f"Mask {mask_pth} does not exist."
        assert os.path.exists(img_pth), f"Image {img_pth} does not exist."

        full_frame = img_as_float32(cv2.imread(full_frame_pth))

        ind = int(os.path.splitext(os.path.basename(face_pth).split("_")[1])[0])
        box = boxes[ind]

        if args.resize_first:
            imgA = full_frame[box[1]:box[3], box[0]:box[2]]
        else:
            imgA = img_as_float32(cv2.imread(img_pth))
        imgB = img_as_float32(cv2.imread(face_pth))
        mask = img_as_float32(cv2.imread(mask_pth))
        shape = imgB.shape

        new = blender(imgA, imgB, mask)
        if args.resize_first:
            full_frame[box[1]:box[3], box[0]:box[2]] = new
        else:
            full_frame[box[1]:box[3], box[0]:box[2]] = cv2.resize(new, (box[2]-box[0], box[3]-box[1]), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(face_pth.replace("/faces/", "/full_frames/").replace(".png", ".jpg"), img_as_ubyte(np.clip(full_frame, 0, 1)))

        if args.save_images:
            if new.shape!=shape:
                new = cv2.resize(new, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(face_pth.replace("/faces/", "/images/"), img_as_ubyte(np.clip(new,0,1)))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_dir", type=str, help="Path to subject folder.")
    parser.add_argument("--exp_name", type=str, default="demo", help="Experiment sub-folder")
    parser.add_argument("--resize_first", action="store_true", help="If specified, first resize image, then blend, else reversely")
    parser.add_argument("--save_images", action="store_true", help="If specified, save the cropped blended images, apart from the full frames")
    parser.add_argument("--method", type=str, default="pyramid", choices = ["copy_paste", "pyramid", "poisson"], help="Blending method")
    parser.add_argument("--n_levels", type=int, default=4, help="Number of levels of the laplacian pyramid, if pyramid blending is used")
    parser.add_argument("--n_levels_copy", type=int, default=0, help="Number of levels at the top of the laplacian pyramid to copy from image A")
    args = parser.parse_args()

    # Figure out the device
    device = "cuda"

    # Get the path of each aligned face image
    faces_a_dir = os.path.join(args.subject_dir, args.exp_name, "faces_aligned")
    face_a_paths = utils.get_faces_a_paths(faces_a_dir)

    faces = unalign(face_a_paths, args)
    utils.save_faces(face_a_paths, faces, args)

    # Use the unaligned face paths 
    faces_path = os.path.join(args.subject_dir, args.exp_name, "faces")
    face_paths = utils.get_image_paths(faces_path)

    blender = Blend(method=args.method, n_levels=args.n_levels, n_levels_copy=args.n_levels_copy, device=device)
    blend_and_save_images(face_paths, blender, args)
