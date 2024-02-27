import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from spectre.config import cfg as spectre_cfg
from spectre.src.models.FLAME import FLAME
from spectre.src.utils.util import batch_orth_proj
from spectre.src.utils.renderer import SRenderY


def isolate_params(data, cfg, full_pose=False):
    """ Input with shape [BATCH, TOTAL_N_FLAME].
    Should return ([BATCH, N_SHAPE], [BATCH, N_POSE], [BATCH, N_EXP]). """
    shape = torch.zeros((cfg.batch_size, cfg.n_shape,), device=data.device)

    # POSE: {head, jaw} x {pitch, yaw, roll}
    if full_pose:
        pose = data[:, :cfg.n_pose]
    else:
        pose = torch.zeros((cfg.batch_size, cfg.n_pose,), device=data.device)
        pose[:, 3] = data[:, 3]
    exp = data[:, cfg.n_pose:]
    return shape, pose, exp


def render_blendshapes(blendshapes, resolution=256, full_pose=False, cams=None):
    # Expects [1, SEQ_LEN, 56]
    # Set up FLAME model
    cfg = spectre_cfg.model
    max_len = 50
    split = blendshapes.shape[1] > max_len
    camera_t = 6.5

    renderer = SRenderY(resolution, obj_filename="spectre/data/head_template.obj").cuda()

    # Set the maximum batch size to max_len. If a sequence is larger, it has to be
    # split into max_len-length subsequences in order to render, for memory saving.
    if not split:
        cfg.batch_size = blendshapes.shape[1]
        flamelayer = FLAME(cfg).to(blendshapes.device)
        shape, pose, exp = isolate_params(blendshapes[0], cfg, full_pose=full_pose)
        # FLAME returns (vertices, landmarks2d, landmarks3d)
        vertices, landmarks = flamelayer(shape, exp, pose)[:2]

        if cams is None:
            cam = torch.zeros((blendshapes.shape[1], 3), dtype=torch.float32).cuda()
            cam[..., 0] = camera_t
        else:
            cam = cams

        # Projection
        landmarks = batch_orth_proj(landmarks, cam)[:, :, :2]
        landmarks[:, :, 1:] = -landmarks[:, :, 1:]
        trans_verts = batch_orth_proj(vertices, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        trans_verts[:, :, 2] = trans_verts[:, :, 2] + 10

        video = renderer.render_shape(vertices, trans_verts)[0].permute(0, 2, 3, 1).cpu().numpy()
        landmarks = landmarks.cpu()

    else:
        videos = []
        landmarks = []
        for i in range(0, blendshapes.shape[1], max_len):
            item = blendshapes[:, i:(i + max_len), :]
            cfg.batch_size = item.shape[1]
            flamelayer = FLAME(cfg).to(blendshapes.device)
            shape, pose, exp = isolate_params(item[0], cfg, full_pose=full_pose)
            ret = flamelayer(shape, exp, pose)
            vertices, landmarks_chunk = ret[:2]

            if cams is None:
                cam = torch.zeros((cfg.batch_size, 3), dtype=torch.float32).cuda()
                cam[..., 0] = camera_t
            else:
                cam = cams[:cfg.batch_size]

            # Projection
            landmarks_chunk = batch_orth_proj(landmarks_chunk, cam)[:, :, :2]
            landmarks_chunk[:, :, 1:] = -landmarks_chunk[:, :, 1:]
            trans_verts = batch_orth_proj(vertices, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
            trans_verts[:, :, 2] = trans_verts[:, :, 2] + 10

            video = renderer.render_shape(vertices, trans_verts)[0].permute(0, 2, 3, 1).cpu().numpy()
            landmarks_chunk = landmarks_chunk.cpu()
            videos.append(video)
            landmarks.append(landmarks_chunk)

        video = np.vstack(videos)
        landmarks = torch.cat([l for l in landmarks])

    video = (255 * video).astype(np.uint8)
    landmarks = landmarks * resolution//2 + resolution//2
    return video, landmarks
