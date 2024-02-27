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
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from spectre.src.models.encoders import PerceptualEncoder
from spectre.src.utils.renderer import SRenderY
from spectre.src.models.encoders import ResnetEncoder
from spectre.src.models.FLAME import FLAME, FLAMETex
from spectre.src.utils import util
from spectre.src.utils.tensor_cropper import transform_points


class SPECTRE(nn.Module):
    def __init__(self, config=None, device="cuda"):
        super().__init__()
        self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)


    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size).to(self.device)


    def _create_model(self, model_cfg):
        # Set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.param_dict = {i: model_cfg.get("n_" + i) for i in model_cfg.param_list}

        # Encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)

        self.E_expression = PerceptualEncoder(model_cfg).to(self.device)

        # Decoders
        self.flame = FLAME(model_cfg).to(self.device)
        self.nmfc = self.flame.v_template.view(1, -1, 3)
        self.nmfc = (self.nmfc - self.nmfc.min(dim=1, keepdim=True)[0]) / (self.nmfc.max(dim=1, keepdim=True)[0] - self.nmfc.min(dim=1, keepdim=True)[0])
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)

        # Resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)

            if "state_dict" in checkpoint.keys():
                self.checkpoint = checkpoint["state_dict"]
            else:
                self.checkpoint = checkpoint

            processed_checkpoint = {}
            processed_checkpoint["E_flame"] = {}
            processed_checkpoint["E_expression"] = {}
            if "deca" in list(self.checkpoint.keys())[0]:
                for key in self.checkpoint.keys():
                    k = key.replace("deca.", "")
                    if "E_flame" in key:
                        processed_checkpoint["E_flame"][k.replace("E_flame.", "")] = self.checkpoint[key]
                    elif "E_expression" in key:
                        processed_checkpoint["E_expression"][k.replace("E_expression.", "")] = self.checkpoint[key]
                    else:
                        pass
            else:
                processed_checkpoint = self.checkpoint

            self.E_flame.load_state_dict(processed_checkpoint["E_flame"], strict=True)

            try:
                self.E_expression.load_state_dict(processed_checkpoint["E_expression"], strict=True)
            except Exception as e:
                print(f"Missing keys {e} in expression encoder weights. If starting training from scratch this is normal.")
        else:
            raise(f"please check model path: {model_path}")

        self.E_flame.eval()
        self.E_expression.eval()
        self.E_flame.requires_grad_(False)


    def decompose_code(self, code, num_dict):
        """ Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ["shape", "tex", "exp", "pose", "cam", "light"]
        """
        code_dict = {}
        start = 0

        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[..., start:end]
            start = end
            if key == "light":
                dims_ = code_dict[key].ndim -1 # (to be able to handle batches of videos)
                code_dict[key] = code_dict[key].reshape(*code_dict[key].shape[:dims_], 9, 3)
        return code_dict


    def encode(self, images):
        with torch.no_grad():
            parameters = self.E_flame(images)

        codedict = self.decompose_code(parameters, self.param_dict)
        deca_exp = codedict["exp"].clone()
        deca_jaw = codedict["pose"][...,3:].clone()

        codedict["images"] = images

        codedict["exp"], jaw = self.E_expression(images)
        codedict["pose"][..., 3:] = jaw

        return codedict, deca_exp, deca_jaw


    def decode(
        self,
        codedict,
        return_vis=True
    ):
        images = codedict["images"]

        is_video_batch = images.ndim == 5
        if is_video_batch:
            B, T, C, H, W = images.shape
            images = images.view(B*T, C, H, W)
            codedict_ = codedict 
            codedict = {}
            for key in codedict_.keys():
                codedict[key] = codedict_[key].view(B*T, *codedict_[key].shape[2:])

        batch_size = images.shape[0]

        # Decode
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict["shape"],
            expression_params=codedict["exp"],
            pose_params=codedict["pose"]
        )
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict["tex"]).detach()
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)

        # Projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict["cam"])[:, :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict["cam"])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]
        trans_verts = util.batch_orth_proj(verts, codedict["cam"])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        trans_verts[:, :, 2] = trans_verts[:, :, 2] + 10

        opdict = {
            "verts": verts,
            "trans_verts": trans_verts,
            "landmarks2d": landmarks2d,
            "landmarks3d": landmarks3d
        }

        if self.cfg.model.use_tex:
            opdict["albedo"] = albedo

        if is_video_batch:
            for key in opdict.keys():
                opdict[key] = opdict[key].view(B, T, *opdict[key].shape[1:])

        if return_vis:
            # Render shapes
            shape_images, pos_mask = self.render.render_shape(
                verts,
                trans_verts
            )
            # Render NMFCs
            nmfc_images = self.render.render_colors(trans_verts, self.nmfc, pos_mask)

            visdict = {
                "shape_images": shape_images,
                "nmfc_images": nmfc_images
            }

            if is_video_batch:
                for key in visdict.keys():
                    visdict[key] = visdict[key].view(B, T, *visdict[key].shape[1:])

            return opdict, visdict

        else:
            return opdict
