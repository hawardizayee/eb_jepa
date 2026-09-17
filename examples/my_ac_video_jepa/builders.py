"""Config and model construction for notebooks."""

from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import init_data
from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.training_utils import load_config

from architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)

from jepa import JEPA, JEPAProbe
from state_decoder import MLPXYHead
from losses import SquareLossSeq,VC_IDM_Sim_Regularizer

CFG_DIR = Path(__file__).resolve().parent / "cfgs"


def build_train_cfg(cfg_path=None):
    """cfgs/train.yaml -> cfg (dot notation) + the train/val data it describes."""
    cfg = load_config(cfg_path or CFG_DIR / "train.yaml")
    cfg.data.num_workers = 0 
    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )
    return cfg, loader, val_loader, data_config


def build_eval_cfg(cfg_path=None):
    """cfgs/eval.yaml -> eval_cfg + env_config + an env_creator (as main.py builds it)."""
    eval_cfg = load_config(cfg_path or CFG_DIR / "eval.yaml")
    # eval.yaml only carries data overrides; base cfg is two_rooms/data_config.yaml
    _, _, env_config = init_data(
        env_name=eval_cfg.data.env_name,
        cfg_data=OmegaConf.to_container(eval_cfg.data, resolve=True)
    )
    env_kwargs = OmegaConf.to_container(eval_cfg.env, resolve=True)   # n_allowed_steps, level

    def env_creator():
        return DotWall(config=env_config, **env_kwargs)

    return eval_cfg, env_config, env_creator


def build_plan_cfg(cfg_path=None, logging_cfg=None):
    """cfgs/planning_mppi.yaml -> plan_cfg (DictConfig, what GCAgent expects)."""
    plan_cfg = load_config(cfg_path or CFG_DIR / "planning_mppi.yaml")
    if logging_cfg is not None:     # main.py mirrors train.yaml's logging into plan_cfg
        plan_cfg.logging = OmegaConf.create(dict(logging_cfg))
    return plan_cfg


def build_model(cfg, data_config, normalizer, device="cpu"):
    encoder = ImpalaEncoder(
    width=1,
    stack_sizes=(16, cfg.model.henc, cfg.model.dstc),   #(16, henc = 32, dstc = 32)
    num_blocks=2,
    dropout_rate=None,
    layer_norm=False,
    input_channels=cfg.model.dobs,
    final_ln=True,
    mlp_output_dim=512,
    input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
)
    aencoder = nn.Identity()
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln   
    )

    projector = Projector(
        f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}"
    )

    test_input = torch.rand(
        (
            1,
            cfg.model.dobs,         # 2
            1,
            data_config.img_size,   # 65 
            data_config.img_size    # 65
        )
    )
    test_output = encoder(test_input) 
    _, f, _, h, w = test_output.shape

    idm = InverseDynamicsModel(
        state_dim=h
        * w
        * (projector.out_dim if cfg.model.regularizer.idm_after_proj else f),   # f 
        hidden_dim=256,
        action_dim=2
    ).to(device)

    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,              # 8 
        std_coeff=cfg.model.regularizer.std_coeff,              # 16 
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,          # 12
        idm_coeff= cfg.model.regularizer.get("idm_coeff", 0.1), # 1
        idm = idm,  # InverseDynamicsModel
        first_t_only=cfg.model.regularizer.get("first_t_only"), # False
        projector=projector,    # None 
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples, # False
        idm_after_proj=cfg.model.regularizer.idm_after_proj,         # False
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj      # False
    )
    ploss = SquareLossSeq()

    jepa = JEPA(
        encoder,     # ImpalaEncoder
        aencoder,    # Identity()
        predictor,   # RNNPredictor
        regularizer, # VC_IDM_Sim_Regularizer
        ploss        # SquareLossSeq
        )


    #--PROBER
    xy_head = MLPXYHead(
        input_shape= test_output.shape[1],  # f = 512
        normalizer=normalizer
    ).to(device)
    xy_prober = JEPAProbe(
        jepa=jepa,
        head=xy_head,
        hcost=nn.MSELoss()
    )

    return jepa, xy_prober
