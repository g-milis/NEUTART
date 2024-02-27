import argparse
import os

import cv2
from moviepy.editor import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_path", type=str, nargs="+", default=".", help="Path to saved frame images.")
    parser.add_argument("--out_path", type=str, default=".", help="Path for saving the video.")
    parser.add_argument("--fps", type=float, default=25, help="Desired fps.")
    parser.add_argument("--wav", type=str, default=None, help="Path to .wav file that contains audio.")
    args = parser.parse_args()

    for root, _, fnames in sorted(os.walk(args.imgs_path[0])):
        if len(fnames)==0:
            continue
        for name in sorted(fnames):
            im = cv2.imread(os.path.join(root, name))
            w, h = im.shape[1], im.shape[0]
            break
        break

    video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    print("Converting frames to video...")
    for root, _, fnames in sorted(os.walk(args.imgs_path[0])):
        for name in sorted(fnames):
            im = cv2.imread(os.path.join(root, name))
            video.write(im)

    cv2.destroyAllWindows()
    video.release()

    if args.wav is not None:
        video = VideoFileClip(args.out_path)
        video_audio = AudioFileClip(args.wav)
        # Join video with audio and write back
        video = video.set_audio(video_audio)
        os.remove(args.out_path)
        video.write_videofile(args.out_path, logger=None)

    print(f"Video saved in: {args.out_path}")
