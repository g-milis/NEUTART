import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audiovisual.utils.render import isolate_params
from spectre.config import cfg
from spectre.src.utils.renderer import SRenderY
from spectre.src.models.FLAME import FLAME
from spectre.src.utils.util import batch_orth_proj
import utils.lipread as lipread_utils


class AudiovisualLoss(nn.Module):
    def __init__(self, train_config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.gradient_loss = torch.nn.MSELoss(reduction="none")

        losses = train_config["losses"]
        self.audiovisual = "b" in losses
        self.use_gradient = "g" in losses
        self.use_flow = "f" in losses
        self.use_lipreading = "l" in losses

        # Lip reading loss
        if self.use_lipreading:
            self.cosine_similarity = nn.CosineSimilarity()
            self.resolution = 224      
            self.renderer = SRenderY(self.resolution, obj_filename="spectre/data/head_template.obj").cuda()
            self.lip_reader = lipread_utils.get_lip_reader()
            self.flame_config = cfg.model
            
            # Set up FLAME model
            self.flame_config.batch_size = 1
            self.flamelayer = FLAME(self.flame_config).to("cuda")
            
            # Albedo is an image: [3, 512, 512]
            self.albedo = torch.tensor(
                plt.imread("spectre/data/mean_texture.jpg")
            ).float().permute(2, 0, 1).to("cuda")


    def differentiable_render(self, blendshapes, pose, cams):
        # Expects [N, 56]
        max_len = 130
        split = blendshapes.shape[0] > max_len

        # Set the maximum batch size to max_len. If a sequence is larger, it has to be
        # split into max_len-length subsequences in order to render, for memory saving.
        if not split:
            self.flamelayer.batch_size = blendshapes.shape[0]
            self.flame_config.batch_size = blendshapes.shape[0]

            shape, _, exp = isolate_params(blendshapes, self.flame_config)
            vertices, landmarks = self.flamelayer(shape, exp, pose)[:2]
            cam = cams
            albedo = self.albedo.repeat(blendshapes.shape[0], 1, 1, 1)

            # Projection
            landmarks = batch_orth_proj(landmarks, cam)[..., :2]
            landmarks[..., 1:] = -landmarks[..., 1:]
            trans_verts = batch_orth_proj(vertices, cam)
            trans_verts[..., 1:] = -trans_verts[..., 1:]

            video = self.renderer(vertices, trans_verts, albedo)["images"].permute(0, 2, 3, 1)
            landmarks = landmarks
        else:
            videos = []
            landmarks = []
            for i in range(0, blendshapes.shape[0], max_len):
                item = blendshapes[i:(i + max_len), :]
                pose_item = pose[i:(i + max_len), :]
                         
                self.flamelayer.batch_size = item.shape[0]
                self.flame_config.batch_size = item.shape[0]

                shape, _, exp = isolate_params(item, self.flame_config)
                vertices, landmarks_chunk = self.flamelayer(shape, exp, pose_item)[:2]
                cam = cams[i:(i + max_len), :]
                albedo = self.albedo.repeat(item.shape[0], 1, 1, 1)

                # Projection
                landmarks_chunk = batch_orth_proj(landmarks_chunk, cam)[..., :2]
                landmarks_chunk[..., 1:] = -landmarks_chunk[..., 1:]
                trans_verts = batch_orth_proj(vertices, cam)
                trans_verts[..., 1:] = -trans_verts[..., 1:]

                video = self.renderer(vertices, trans_verts, albedo)["images"].permute(0, 2, 3, 1)
                landmarks_chunk = landmarks_chunk
                videos.append(video)
                landmarks.append(landmarks_chunk)

            video = torch.cat([v for v in videos])
            landmarks = torch.cat([l for l in landmarks])
        return video, landmarks

    
    def lipreading_loss(self, mouths_gold, mouths_pred):
        """ Expects videos already cut at the mouth. """
        # Convert to [B, L, H, W]
        mouths_gold = mouths_gold.unsqueeze(0)
        mouths_pred = mouths_pred.unsqueeze(0)

        lip_features_gold = self.lip_reader.model.encoder(
            mouths_gold,
            None,
            extract_resnet_feats=True
        ).squeeze()
        lip_features_pred = self.lip_reader.model.encoder(
            mouths_pred,
            None,
            extract_resnet_feats=True
        ).squeeze()

        similarity = self.cosine_similarity(lip_features_gold, lip_features_pred)
        return 1 - similarity.mean()

    
    def gradient_1st(self, x):
        """ Calculate the temporal derivative MSE.
        x: [B, L, C] """
        x_minus_1 = x[..., 1:]
        x_ = x[..., :-1]
        loss = torch.mean(self.gradient_loss(x_, x_minus_1), dim=-1)
        return loss.mean()


    def compute_flow_loss(self, gold, pred):
        """ Calculate the flow MSE between gold and predicted.
        tensor inputs: [B, L, C] """
        gold_minus_1 = gold[..., 1:]
        gold_ = gold[..., :-1]
        pred_minus_1 = pred[..., 1:]
        pred_ = pred[..., :-1]
        loss = self.mse_loss((gold_ - gold_minus_1), (pred_ - pred_minus_1))
        return loss


    def forward(self, inputs, predictions):
        # Loss weights
        (
            w_energy,
            w_pitch,
            w_duration,
            w_mel,
            w_pose,
            w_expression,
            w_flow,
            w_gradient
        ) = 1, 1, 1, 1, 1, 1, 1, 1

        w_lipreading = 1

        mel_targets = inputs["mels"]
        pitch_targets = inputs["pitches"]
        energy_targets = inputs["energies"]
        duration_targets = inputs["durations"]
        if self.audiovisual:
            blendshape_targets = inputs["blendshapes"]
            if self.use_lipreading:
                mouths_gold = inputs["mouths"]
                cams = inputs["cams"]

        mel_predictions = predictions["mels"]
        pitch_predictions = predictions["pitches"]
        energy_predictions = predictions["energies"]
        log_duration_predictions = predictions["log_durations"]
        if self.audiovisual:
            blendshape_predictions = predictions["blendshapes"]
        src_masks = predictions["src_masks"]
        mel_masks = predictions["mel_masks"]
        mel_lens = predictions["mel_lens"]

        # Invert True to False (masks is True when value is ignored)
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        if self.audiovisual:
            blendshape_targets.requires_grad = False

        # Mask predicted sequences before losses
        pitch_predictions = pitch_predictions.masked_select(src_masks)
        pitch_targets = pitch_targets.masked_select(src_masks)
        energy_predictions = energy_predictions.masked_select(src_masks)
        energy_targets = energy_targets.masked_select(src_masks)
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # Calculate TTS losses
        # First, decoder losses on the spectrogram
        mel_loss = w_mel * self.mae_loss(mel_predictions, mel_targets)
        # Then, encoder losses from the variance adaptor modules
        pitch_loss = w_pitch * self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = w_energy * self.mse_loss(energy_predictions, energy_targets)
        duration_loss = w_duration * self.mse_loss(log_duration_predictions, log_duration_targets)

        # blendshapes: [B, L, N_POSE + N_EXP]
        # Isolate pose from expression to apply different loss weights
        blendshape_loss = torch.tensor([0], device="cuda")
        velocity_loss = torch.tensor([0], device="cuda")
        flow_loss = torch.tensor([0], device="cuda")
        lipreading_loss = torch.tensor([0], device="cuda")
        exp_reg_loss = torch.tensor([0], device="cuda")
        if self.audiovisual:
            pose_predictions = blendshape_predictions[:, :, :6]
            pose_targets = blendshape_targets[:, :, :6]
            expression_predictions = blendshape_predictions[:, :, 6:]
            expression_targets = blendshape_targets[:, :, 6:]

            pose_predictions = pose_predictions.masked_select(mel_masks.unsqueeze(-1))
            pose_targets = pose_targets.masked_select(mel_masks.unsqueeze(-1))
            expression_predictions = expression_predictions.masked_select(mel_masks.unsqueeze(-1))
            expression_targets = expression_targets.masked_select(mel_masks.unsqueeze(-1))
            
            # Expression and pose MSE loss
            pose_loss = w_pose * self.mse_loss(pose_predictions, pose_targets)
            expression_loss = w_expression * self.mse_loss(expression_predictions, expression_targets)
            blendshape_loss = pose_loss + expression_loss

            # FLow loss
            if self.use_flow:
                flow_loss = w_flow * self.compute_flow_loss(blendshape_targets, blendshape_predictions)
                
            # Velocity loss
            if self.use_gradient:
                velocity_loss = w_gradient * self.gradient_1st(blendshape_predictions)

            # Lipreading loss
            if self.use_lipreading:
                # Expression regularization loss
                reg = torch.sum((expression_predictions - expression_targets) ** 2, dim=-1) / 2
                weight_vector = torch.ones_like(reg).cuda()
                weight_vector[reg > 40] = 2e-3
                weight_vector[reg < 40] = 1e-3
                exp_reg_loss = torch.mean(weight_vector * reg)

                lipreading_loss = 0
                for i, video in enumerate(mouths_gold):
                    blendshapes_pred_unpadded = blendshape_predictions[i, :mel_lens[i], :]
                    poses_unpadded = blendshape_targets[i, :mel_lens[i], :6]
                    blendshapes_interp = torch.nn.functional.interpolate(
                        blendshapes_pred_unpadded.T.unsqueeze(0),
                        size=video.shape[0],
                        mode="linear"
                    ).squeeze().T
                    pose_interp = torch.nn.functional.interpolate(
                        poses_unpadded.T.unsqueeze(0),
                        size=video.shape[0],
                        mode="linear"
                    ).squeeze().T

                    n_frames = 10
                    max_idx = video.shape[0] - n_frames - 1
                    # Randomly select a starting row index within the valid range
                    try:
                        idx = np.random.randint(0, max_idx)
                    except:
                        idx = 0
                    selected = blendshapes_interp[idx:(idx + n_frames)]
                    pose_selected = pose_interp[idx:(idx + n_frames)]
                    cams_selected = cams[i][idx:(idx + n_frames)]
                    cams_selected = torch.from_numpy(cams_selected).float().to("cuda", non_blocking=True)

                    cams_selected = torch.zeros((pose_selected.shape[0], 3), device="cuda")
                    cams_selected[:, 0] = 8.5

                    video, lmks = self.differentiable_render(selected, pose_selected, cams_selected)
                    lmks = lmks * self.resolution//2 + self.resolution//2
                    mouths_pred = lipread_utils.cut_mouth(video, lmks)

                    mouths_gold_selected = torch.from_numpy(
                        mouths_gold[i][idx:(idx + n_frames)]
                    ).float().to("cuda", non_blocking=True)
                    lr = w_lipreading * self.lipreading_loss(mouths_gold_selected, mouths_pred)
                    lipreading_loss += lr


                lipreading_loss /= len(mouths_gold)

        total_loss = (
            mel_loss \
            + duration_loss \
            + pitch_loss + energy_loss \
            + blendshape_loss \
            + velocity_loss \
            + flow_loss \
            + lipreading_loss \
            + exp_reg_loss
       )
        return {
            "total_loss": total_loss,
            "mel_loss": mel_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "duration_loss": duration_loss,
            "blendshape_loss": blendshape_loss,
            "velocity_loss": velocity_loss,
            "flow_loss": flow_loss,
            "lipreading_loss": lipreading_loss,
            "exp_reg_loss": exp_reg_loss
        }
