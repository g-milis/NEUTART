"""
Default config for SPECTRE - adapted from DECA
"""

import os

from yacs.config import CfgNode as CN


cfg = CN()
cfg.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", ".."))
cfg.device = "cuda"
cfg.pretrained_modelpath = "spectre/pretrained/spectre_model.tar"

# ---------------------------------------------------------------------------- #
# Options for FLAME and from original DECA
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.project_dir, "data" , "head_template.obj")
cfg.model.flame_model_path = os.path.join(cfg.project_dir, "data", "generic_model.pkl")
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.project_dir, "data", "landmark_embedding.npy")
cfg.model.tex_path = os.path.join(cfg.project_dir, "data", "FLAME_albedo_from_BFM.npz")

cfg.model.uv_size = 256
cfg.model.param_list = ["shape", "tex", "exp", "pose", "cam", "light"]
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.batch_size = 4

cfg.model.temporal = True
cfg.model.use_tex = True
cfg.model.backbone = "mobilenetv2" # perceptual encoder backbone


# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.image_size = 224
