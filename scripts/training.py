import argparse
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from .sae_components import SAELightningModule
from .data_module import FastaDataModule

DEFAULT_CFG = {
    "model_name": "facebook/esm2_t33_650M_UR50D",
    "layer_idx": 24,
    "latent_dim": 4096,
    "topk": 20,
    "l1_lambda": 2e-5,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "fasta_dir": "mala/fasta",
    "max_tokens_per_batch": 2048,
    "batch_size_tokens": 2048,  # kept for yaml compatibility
    "devices": 1,
    "precision": 16,
    "accumulate_grad_batches": 1,
    "num_workers": 4,
    "sample_fraction": 1.0,
    "max_steps": 100_000,
    "log_every_n_steps": 50,
    "val_check_interval": 10_000,  # purely to trigger ckpt saving
}


# -----------------------------------------------------------------------------
# helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_cfg(path: str | None) -> dict:
    if path is None:
        return DEFAULT_CFG
    with open(path, "r") as fp:
        user_cfg = yaml.safe_load(fp)
    cfg = {**DEFAULT_CFG, **user_cfg}
    return cfg


# -----------------------------------------------------------------------------
# cli -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train sparse autoencoder over ESM2 states")
    p.add_argument("--config", type=str, default=None, help="yaml with hyper‑params")
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    p.add_argument("--devices", type=int, default=None, help="override GPU count")
    p.add_argument("--precision", type=int, default=None, help="override AMP precision (16/32)")
    return p.parse_args()


# -----------------------------------------------------------------------------
# entry -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = get_args()
    cfg = load_cfg(args.config)

    if args.devices is not None:
        cfg["devices"] = args.devices
    if args.precision is not None:
        cfg["precision"] = args.precision

    # datamodule -----------------------------------------------------------
    dm = FastaDataModule(
        fasta_dir=cfg["fasta_dir"],
        max_tokens_per_batch=cfg["max_tokens_per_batch"],
        tokenizer_name=cfg["model_name"],
        num_workers=cfg["num_workers"],
        sample_fraction=cfg["sample_fraction"],
    )

    # model ----------------------------------------------------------------
    model = SAELightningModule(
        model_name=cfg["model_name"],
        layer_idx=cfg["layer_idx"],
        latent_dim=cfg["latent_dim"],
        topk=cfg["topk"],
        l1_lambda=cfg["l1_lambda"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # logging / callbacks ---------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    tb_logger = TensorBoardLogger("outputs", name="sae")
    ckpt_cb = ModelCheckpoint(
        dirpath="outputs/sae/checkpoints",
        filename="step{step:07d}",
        save_top_k=-1,
        every_n_train_steps=cfg["val_check_interval"],
        save_weights_only=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    rich_bar = RichProgressBar()

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_cb, rich_bar],
        devices=cfg["devices"],
        accelerator="gpu" if cfg["devices"] else "cpu",
        precision=cfg["precision"],
        accumulate_grad_batches=cfg["accumulate_grad_batches"],
        max_steps=cfg["max_steps"],
        log_every_n_steps=cfg["log_every_n_steps"],
        gradient_clip_val=1.0,
        default_root_dir="outputs/sae",
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

