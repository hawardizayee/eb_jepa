import os 
import time 
from pathlib import Path 

import fire 
import torch 
import torch.nn as nn 
from torch.amp import GradScaler 
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import VisionTransformer
import wandb
from omegaconf import OmegaConf

from model import ResNet18,ImageSSL
from lars import LARS
from scheduler import WarmupCosineScheduler
from log_utils import get_logger
from training_utils import (
    get_default_dev_name,
    load_config,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    log_data_info,
    log_epoch,
    log_model_info,
    log_config,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,

)
from dataset import (
    ImageDataset,
    get_train_transforms,
    get_val_transforms
)
from losses import VICRegLoss, BCS
from engine import train_epoch
from eval import LinearProbe, evaluate_linear_probe

logger = get_logger(__name__)

def run(
        fname: str = "cfgs/default.yaml",
        cfg = None,
        folder = None,
        **overrides
):
    """
    Train an Image JEPA (VICReg/BCS) model on CIFAR-10

    Args: 
        fname: Path to YAML config file 
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., optim.epochs=50)
    
    """ 

    # Load config 
    if cfg is None:
        cfg = load_config(fname,overrides if overrides else None)

    # Setup using shared utilities 
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            folder_name = exp_dir.name 
            exp_name = folder_name.rsplit("_seed",1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("image_jepa",cfg)
            exp_dir = get_unified_experiment_dir(
                example_name = "image_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed
            )

    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True,exist_ok=True)
        # Extract exp_name from folder name by removing _seed{seed} suffix
        folder_name = exp_dir.name  # e.g. "resnet_vicreg_seed1"
        exp_name = folder_name.rsplit("_seed", 1)[0] # e.g., "resnet_vicreg"

    wandb_run = setup_wandb(
        project= "eb_jepa",
        config={"example": "image_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir= exp_dir,
        run_name=exp_name,
        tags=["image_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled= cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id")
    )

    logger.info("Loading CIFAR-10 dataset")
    transform = get_train_transforms()

    # Use EBJEPA_DSETS environment variable if set, otherwise fall back to config
    data_dir = os.environ.get("EBJEPA_DSETS")
    logger.info(f"Using data directory: {data_dir}")

    base_train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )

    train_dataset = ImageDataset(base_train_dataset, transform, num_crops=2)

    val_dataset = CIFAR10(
        root=data_dir, train=False, download=True, transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True  # Avoid small batches that cause BatchNorm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers= cfg.data.num_workers,
        pin_memory=True
    )

    log_data_info(
        "CIFAR-10",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset)
    )

    # Initialize model
    logger.info("Initailizing model...")
    if cfg.model.type == "resnet":
        backbone = ResNet18()
        features_dim = backbone.features_dim
    elif cfg.model.type == "vit_s":
        features_dim = 384
        model_kwargs = dict(
            image_size=32,
            patch_size=8,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=6,
            mlp_dim=4 * features_dim
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()
    elif cfg.model.type == "vit_b":
        features_dim = 768
        model_kwargs = dict(
            image_size=32,
            patch_size=8,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=12,
            mlp_dim= 4 * features_dim
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()

    model = ImageSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim
    )

    if not cfg.model.use_projector:
        model.projector = nn.Identity()

    model = model.to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in backbone.parameters())
    projector_params = (
        sum(p.numel() for p in model.projector.parameters())
        if cfg.model.use_projector
        else 0 
    )
    log_model_info(model, {"encoder": encoder_params, "projector": projector_params})

    # Log Configuration
    log_config(cfg)

    # Initialize Linear probe 
    linear_probe = LinearProbe(feature_dim=features_dim,num_classes=10).to(device)

    # Mixed precision setup 
    dtype_map = {"bfloat16": torch.bfloat16, "float16":torch.float16}
    use_amp = cfg.training.get("use_amp",True)
    dtype = dtype_map.get(cfg.training.get("dtype","float16").lower(), torch.float16)
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype=}" if use_amp else f"AMP disabled")

    optimizer = LARS(
        [
            {"params": model.parameters(),"lr":cfg.optim.lr},   # 0.3
            {"params": linear_probe.parameters(),"lr":0.1}
        ],
        weight_decay=cfg.optim.weight_decay,    # 1.0e-4
        eta= 0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,
        momentum=0.9
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.optim.warmup_epochs,
        max_epochs=cfg.optim.epochs,
        base_lr=cfg.optim.lr,
        min_lr=cfg.optim.min_lr,
        warmup_start_lr=cfg.optim.warmup_start_lr
    )

    # Initialize loss function
    if cfg.loss.type == "vicreg":
        loss_fn = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff)
    elif cfg.loss.type == "bcs":
        loss_fn = BCS(lmbd=cfg.loss.lmbd)

    
    # Load checkpoint if requested 
    start_epoch = 0 
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path,model,optimizer,device= device)
        start_epoch = ckpt_info.get("epoch",0)
        if "linear_probe_state_dict" in ckpt_info:
            linear_probe.load_state_dict(ckpt_info["linear_probe_state_dict"])
    

    # Fixed monitor batch (256) for tracking encoder + projector embeddings across epochs
    monitor_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    monitor_views, _ = next(iter(monitor_loader))
    monitor_v1 = monitor_views[0].to(device)
    monitor_v2 = monitor_views[1].to(device)
    f1_history, f2_history, z1_history, z2_history = [], [], [], []

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")
    start_time = time.time()
    use_amp = cfg.training.get("use_amp", True)
    tqdm_silent = cfg.logging.get("tqdm_silent", False)

    for epoch in range(start_epoch, cfg.optim.epochs):
        # Train 
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            linear_probe,
            scaler,
            device,
            epoch,
            loss_fn,
            use_amp,
            dtype,
            tqdm_silent
        )

        # Evaluate linear probe on validation set
        val_acc, val_loss = evaluate_linear_probe(
            model, linear_probe, val_loader, device, use_amp
        )

        # Track encoder (f) + projector (z) embeddings on the fixed monitor batch
        model.eval()
        with torch.no_grad():
            f1_epoch, z1_epoch = model(monitor_v1)
            f2_epoch, z2_epoch = model(monitor_v2)
        f1_history.append(f1_epoch.cpu())
        f2_history.append(f2_epoch.cpu())
        z1_history.append(z1_epoch.cpu())
        z2_history.append(z2_epoch.cpu())
        model.train()

        # Log metrics - dynamically add train_prefix to all train_metrics keys
        log_dict = {"epoch": epoch}
        for key, value in train_metrics.items():
            log_dict[f'train_{key}'] = value 
        log_dict["val_loss"] = val_loss
        log_dict["val_acc"] = val_acc
        log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]

        if wandb_run:
            wandb.log(log_dict)

        # Log progress 
        if epoch % cfg.logging.log_every == 0:
            elapsed = time.time() - start_time
            log_epoch(
                epoch,
                {
                    "loss": train_metrics["loss"],
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                total_epochs=cfg.optim.epochs,
                elapsed_time=elapsed
            )
        
        # Save checkpoint
        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model = model,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            linear_probe_state_dict=linear_probe.state_dict(),
            linear_val_acc=val_acc
        )
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                scaler=scaler,
                linear_probe_state_dict=linear_probe.state_dict(),
                linear_val_acc=val_acc
            )

    logger.info("Training completed!")

    # Save embedding trajectories (encoder + projector, both views) for collapse analysis
    torch.save(
        {
            "f1_history": f1_history,
            "f2_history": f2_history,
            "z1_history": z1_history,
            "z2_history": z2_history,
        },
        exp_dir / "embedding_history.pt",
    )

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(run)