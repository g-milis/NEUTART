import argparse
import os
import sys

import cv2
import numpy as np
from facenet_pytorch import MTCNN, extract_face
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import photorealistic.utils as utils

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def tensor2npimage(image_tensor, imtype=np.uint8):
    # Tesnor in range [0, 255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i], imtype))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, transpose=True):
    if transpose:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_images(images, folder, n_videos, args, suffix=".png"):
    for i in range(len(images)):
        n_frame = f"{n_videos:03d}_{i:04d}"
        save_image(
            images[i],
            os.path.join(args.subject_dir, folder, n_frame + suffix),
            transpose = folder == "images"
        )


def smooth_boxes(boxes, previous_box, args):
    # Check if there are None boxes
    if boxes[0] is None:
        boxes[0] = previous_box
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    boxes = [box[0] for box in boxes]
    # Smooth boxes
    old_boxes = np.array(boxes)
    window_length = min(args.window_length, old_boxes.shape[0])
    if window_length % 2 == 0:
        window_length -= 1
    smooth_boxes = np.concatenate(
        [
            ndimage.median_filter(list(old_boxes[:, i]), size=window_length).reshape((-1,1))
            for i in range(4)
        ],
        1
    )
    # Make boxes square
    for i in range(len(smooth_boxes)):
        offset_w = smooth_boxes[i][2] - smooth_boxes[i][0]
        offset_h = smooth_boxes[i][3] - smooth_boxes[i][1]
        offset_dif = (offset_h - offset_w) / 2
        # width
        smooth_boxes[i][0] = smooth_boxes[i][2] - offset_w - offset_dif
        smooth_boxes[i][2] = smooth_boxes[i][2] + offset_dif
        # height - center a bit lower
        smooth_boxes[i][3] = smooth_boxes[i][3] + args.height_recentre * offset_h
        smooth_boxes[i][1] = smooth_boxes[i][3] - offset_h

    return smooth_boxes


def get_faces(detector, images, previous_box, args):
    ret_faces = []
    ret_boxes = []

    all_boxes = []
    all_imgs = []

    # Get bounding boxes
    for lb in np.arange(0, len(images), args.mtcnn_batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+args.mtcnn_batch_size]]
        boxes, _ = detector.detect(imgs_pil)
        all_boxes.extend(boxes)
        all_imgs.extend(imgs_pil)
    # Temporal smoothing
    boxes = smooth_boxes(all_boxes, previous_box, args)
    # Crop face regions.
    for img, box in zip(all_imgs, boxes):
        face = extract_face(img, box, args.cropped_image_size, args.margin)
        ret_faces.append(face)
        # Find real bbox   (taken from https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/utils/detect_face.py#L358)
        margin = [
            args.margin * (box[2] - box[0]) / (args.cropped_image_size - args.margin),
            args.margin * (box[3] - box[1]) / (args.cropped_image_size - args.margin),
        ]
        raw_image_size = img.size
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]
        ret_boxes.append(box)

    return ret_faces, ret_boxes, boxes[-1]


def detect_and_save_faces(detector, mp4_path, n_videos, args, test_name="images"):
    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    images = []
    previous_box = None

    print(f"Reading {mp4_path}, extracting faces, and saving images")
    os.makedirs(os.path.join(args.subject_dir, "boxes"), exist_ok=True)
    os.makedirs( os.path.join(args.subject_dir, test_name), exist_ok=True)
    for _ in tqdm(range(n_frames)):
        _, image = reader.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(images) < args.filter_length:
            images.append(image)
        # else, detect faces in sequence and create new list
        else:
            face_images, boxes, previous_box = get_faces(detector, images, previous_box, args)
            save_images(tensor2npimage(face_images), test_name, n_videos, args, suffix=".png")

            if args.mode == "test":
                save_images(images, "full_frames", n_videos, args, suffix=".jpg")

                txt_file = os.path.join(args.subject_dir, "boxes", os.path.basename(mp4_path).replace("mp4", "txt"))
                if not os.path.exists(txt_file):
                    vfile = open(txt_file, "a")
                    vfile.write("{} {} fps {} frames\n".format(mp4_path, fps, n_frames))
                    vfile.close()
                for box in boxes:
                    vfile = open(txt_file, "a")
                    np.savetxt(vfile, np.expand_dims(box, 0))
                    vfile.close()

            images = [image]
    # last sequence
    face_images, boxes, _ = get_faces(detector, images, previous_box, args)

    save_images(tensor2npimage(face_images), test_name, n_videos, args, suffix=".png")

    # Save video information
    txt_file = os.path.join(args.subject_dir, "boxes", os.path.basename(mp4_path).replace("mp4", "txt"))
    if not os.path.exists(txt_file):
        vfile = open(txt_file, "a")
        vfile.write(f"{mp4_path} {fps} fps {n_frames} frames - video {n_videos}\n")
        vfile.close()

    if args.mode == "test":
        os.makedirs(os.path.join(args.subject_dir, "full_frames"), exist_ok=True)
        save_images(images, "full_frames", n_videos, args, suffix=".jpg")

        for box in boxes:
            vfile = open(txt_file, "a")
            np.savetxt(vfile, np.expand_dims(box, 0))
            vfile.close()

    reader.release()


def main():
    print("Face detection...")
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Whether the videos are intended for training the renderer or running inference.")
    parser.add_argument("--videos_path", type=str, required=True, help="Path to subject videos.")
    parser.add_argument("--subject_dir", type=str, required=True, help="Path to subject preprocessing folder.")
    parser.add_argument("--mtcnn_batch_size", default=8, type=int, help="The number of frames for face detection.")
    parser.add_argument("--select_largest", action="store_true", help="In case of multiple detected faces, keep the largest (if specified), or the one with the highest probability")
    parser.add_argument("--cropped_image_size", default=256, type=int, help="The size of frames after cropping the face.")
    parser.add_argument("--margin", default=70, type=int, help=".")
    parser.add_argument("--filter_length", default=500, type=int, help="Number of consecutive bounding boxes to be filtered")
    parser.add_argument("--window_length", default=49, type=int, help="savgol filter window length.")
    parser.add_argument("--height_recentre", default=0.0, type=float, help="The amount of re-centring bounding boxes lower on the face.")
    args = parser.parse_args()

    device = "cuda"

    # Store video paths in list.
    mp4_paths = utils.get_video_paths(args.videos_path)
    n_mp4s = len(mp4_paths)
    print(f"Number of videos to process: {n_mp4s}")

    # Initialize the MTCNN face  detector.
    detector = MTCNN(image_size=args.cropped_image_size, select_largest=args.select_largest, margin=args.margin, post_process=False, device=device)

    # Run detection
    n_completed = 0
    for path in mp4_paths:
        detect_and_save_faces(detector, path, n_completed, args)
        n_completed += 1
        print(f"({n_completed}/{n_mp4s}) [SUCCESS]")


if __name__ == "__main__":
    main()
