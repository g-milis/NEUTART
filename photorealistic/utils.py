import os

import cv2
import numpy as np


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg"])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in [".mp4"])


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])


def get_image_paths(dir):
    # Returns list: [path1, path2, ...]
    image_files = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_files.append(path)
    return image_files


def get_video_paths(dir):
    # Returns list of paths to video files
    video_files = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_video_file(fname):
                path = os.path.join(root, fname)
                video_files.append(path)
    return video_files


def get_mats_paths(dir):
    # Returns list: [path1, path2, ...]
    mats_files = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                mats_files.append(path)
    return mats_files


def get_faces_a_paths(dir):
    # Returns list: [path1, path2, ...]
    faces_a_files = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for fname in sorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            faces_a_files.append(path)
    return faces_a_files


def save_faces(face_a_pths, faces, args):
    mkdir(os.path.join(args.subject_dir, args.exp_name, "faces"))
    for face_a_pth, face in zip(face_a_pths, faces):
        cv2.imwrite(face_a_pth.replace("/faces_aligned/", "/faces/"), face)


def transform_points(points, mat):
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points
