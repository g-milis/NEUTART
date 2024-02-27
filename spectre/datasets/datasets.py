# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
from glob import glob

import cv2
import numpy as np
import scipy
import scipy.io
import torch
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset

from . import detectors


# def video2sequence(video_path):
#     videofolder = video_path.split(".")[0]
#     os.makedirs(videofolder, exist_ok=True)
#     video_name = video_path.split("/")[-1].split(".")[0]
#     vidcap = cv2.VideoCapture(video_path)
#     success,image = vidcap.read()
#     count = 0
#     imagepath_list = []
#     while success:
#         imagepath = f"{videofolder}/{video_name}_frame{count:04d}.jpg"
#         cv2.imwrite(imagepath, image)     # save frame as JPEG file
#         success,image = vidcap.read()
#         count += 1
#         imagepath_list.append(imagepath)
#     print(f"video frames are stored in {videofolder}")
#     return imagepath_list


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector="fan", device="cuda"):
        """
            testpath: folder, imagepath_list, image path, video path
        """
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + "/*.jpg") +  glob(testpath + "/*.png") + glob(testpath + "/*.bmp") \
                                    + glob(testpath + "/*/*.jpg") +  glob(testpath + "/*/*.png") + glob(testpath + "/*/*.bmp")
        elif os.path.isfile(testpath) and (testpath[-3:] in ["jpg", "png", "bmp"]):
            self.imagepath_list = [testpath]
        # elif os.path.isfile(testpath) and (testpath[-3:] in ["mp4", "csv", "vid", "ebm"]):
        #     self.imagepath_list = video2sequence(testpath)
        else:
            print(f"please check the test path: {testpath}")
            exit()
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == "fan":
            self.face_detector = detectors.FAN(device=device)
        else:
            print(f"please check the detector: {face_detector}")
            exit()


    def __len__(self):
        return len(self.imagepath_list)


    def bbox2point(self, left, right, top, bottom, type="bbox"):
        """ bbox from detector and landmarks are different """
        if type=="kpt68":
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=="bbox":
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center


    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]

        image = np.array(imread(imagepath))
        original_size = (image.shape[1], image.shape[0])
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        if self.iscrop:
            # Default to FAN detection, since SPECTRE has been trained with it
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 4:
                print("no face detected! run original image")
                left = 0; right = h - 1; top = 0; bottom = w - 1
            else:
                left = bbox[0]; right = bbox[2]
                top = bbox[1]; bottom = bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size * self.scale)
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1]+size/2], [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)

        image = image / 255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)
        return {
            "image": torch.tensor(dst_image).float(),
            "imagepath": imagepath,
            "tform": tform,
            "original_size": original_size
        }
