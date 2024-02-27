import argparse
import os
import sys

import cv2
import numpy as np
import torch
from skimage import img_as_float32, img_as_ubyte, io
from skimage.measure import label
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from photorealistic.postprocessing.image_blender import SoftErosion
from photorealistic.preprocessing.segmentation.simple_unet import UNet
import photorealistic.utils as utils


def save_results(args, image_pths, masks):
    # Make dirs
    mask_pths = [p.replace("/images/", "/masks/").replace(".png", ".jpg") for p in image_pths]
    out_paths = set(os.path.dirname(mask_pth) for mask_pth in mask_pths)
    for out_path in out_paths:
        utils.mkdir(out_path)
        if args.mode == "train":
            utils.mkdir(out_path.replace("/masks", "/faces"))
    for mask, image_pth in zip(masks, image_pths):
        image = cv2.imread(image_pth)
        cv2.imwrite(image_pth.replace("/images/", "/masks/").replace(".png", ".jpg"), img_as_ubyte(mask))
        if args.mode == "train":
            cv2.imwrite(image_pth.replace("/images/", "/faces/"), image * mask)


def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest)
    return labels_max


def get_face_masks(img_paths, predictor, smoother, device):
    masks = []
    prev_mask = None
    for i in tqdm(range(len(img_paths))):
        img = img_as_float32(io.imread(img_paths[i]))

        # convert to torch.tensor, change position of channel dim, and add batch dim
        im_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
        # predict face mask
        pred = predictor(im_tensor)  # 3-channel image for 3-wise segmentation (background, face, hair)
        mask = (pred.argmax(1, keepdim=True) == 1)
        _, mask = smoother(mask)   # soft erosion

        # convert to single-channel image
        mask = mask.squeeze(0).permute(1,2,0).cpu().numpy()

        if True in mask:
            # keep only the largest connected component if more than one found
            mask = getLargestCC(mask)
            prev_mask = mask
            masks.append(mask)
        else:
            print("No face mask detected, using previous mask")
            masks.append(prev_mask)

    return masks


def main():
    print("Face segmentation...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_dir", type=str, help="Path to subject folder.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Whether the videos are intended for training the renderer or running inference.")
    args = parser.parse_args()

    # Figure out the device
    device = "cuda"

    # Load pretrained face segmenter
    segmenter_path = "photorealistic/preprocessing/segmentation/lfw_figaro_unet_256_2_0_segmentation_v1.pth"
    checkpoint = torch.load(segmenter_path)
    predictor = UNet(n_classes=3,feature_scale=1).to(device)
    predictor.load_state_dict(checkpoint["state_dict"])
    smooth_mask = SoftErosion(kernel_size=21, threshold=0.6).to(device)

    # Get the path of each image.
    images_dir = os.path.join(args.subject_dir, "images")
    image_paths = utils.get_image_paths(images_dir)

    masks = get_face_masks(image_paths, predictor, smooth_mask, device)
    save_results(args, image_paths, masks)


if __name__=="__main__":
    main()
