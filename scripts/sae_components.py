import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s - %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# ESM wrapper
# ──────────────────────────────────────────────────────────────────────────────
class ESM2Wrapper(nn.Module):
    """Frozen ESM-2 that returns hidden states from a chosen layer."""
    def __init__(self, model_name: str, layer_idx: int):
        super().__init__()
        cfg: AutoConfig = AutoConfig.from_pretrained(model_name,
                                                     output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=cfg)
        self.layer_idx = layer_idx
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.d_model = self.model.config.hidden_size
        logger.info("loaded %s — d_model=%d, frozen", model_name, self.d_model)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask
                          ).hidden_states[self.layer_idx]  # [B, L, d]

# ──────────────────────────────────────────────────────────────────────────────
# SAE core
# ──────────────────────────────────────────────────────────────────────────────
def topk_signed(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= x.size(-1):
        return x
    vals, idx = torch.topk(x.abs(), k, dim=-1)
    mask = torch.zeros_like(x, dtype=torch.bool).scatter(-1, idx, True)
    return x * mask


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, topk: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        self.topk = topk
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = topk_signed(self.encoder(x), self.topk)
        return self.decoder(a), a

    @staticmethod
    @torch.no_grad()
    def dead_ratio(acts: torch.Tensor, eps: float = 1e-5) -> float:
        rms = (acts ** 2).mean(dim=0).sqrt()
        return (rms < eps).float().mean().item() * 100

# ──────────────────────────────────────────────────────────────────────────────
# Lightning glue
# ──────────────────────────────────────────────────────────────────────────────
class SAELightningModule(pl.LightningModule):
    def __init__(self,
                 model_name: str = "facebook/esm2_t33_650M_UR50D",
                 layer_idx: int = 24,
                 latent_dim: int = 4096,
                 topk: int = 20,
                 l1_lambda: float = 2e-5,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.esm = ESM2Wrapper(model_name, layer_idx)
        self.sae = SparseAutoencoder(self.esm.d_model, latent_dim, topk)

        # global activation tracking
        self.register_buffer("ever_act_train",
                             torch.zeros(latent_dim, dtype=torch.bool))
        self.register_buffer("ever_act_val",
                             torch.zeros(latent_dim, dtype=torch.bool))
        self.activation_eps = 1e-5

    # ── training ──────────────────────────────────────────────────────────
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        h = self.esm(batch["input_ids"], batch["attention_mask"])
        tokens = h[batch["attention_mask"].bool()]          # [N, d]
        x_hat, acts = self.sae(tokens)

        recon = F.mse_loss(x_hat, tokens)
        spars = acts.abs().mean()
        loss = recon + self.hparams.l1_lambda * spars

        # global activation update
        self.ever_act_train |= (acts.abs().max(dim=0).values >
                                self.activation_eps)

        dead_pct = self.sae.dead_ratio(acts)
        self.log_dict({"train/recon": recon,
                       "train/sparsity": spars,
                       "train/dead_%": dead_pct,
                       "train/loss": loss},
                      prog_bar=True, on_step=True, on_epoch=False)
        return loss

    # ── validation ───────────────────────────────────────────────────────
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        with torch.no_grad():
            h = self.esm(batch["input_ids"], batch["attention_mask"])
            tokens = h[batch["attention_mask"].bool()]
            x_hat, acts = self.sae(tokens)
            recon = F.mse_loss(x_hat, tokens)
            self.ever_act_val |= (acts.abs().max(dim=0).values >
                                  self.activation_eps)
            self.log("val/recon", recon, prog_bar=True,
                     on_step=False, on_epoch=True)

    # ── optimiser ────────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.sae.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                         T_max=100_000)
        return {"optimizer": opt, "lr_scheduler": sch}

    # ── summary ──────────────────────────────────────────────────────────
    def on_train_end(self):
        total = self.hparams.latent_dim
        train_alive = int(self.ever_act_train.sum())
        val_alive = int(self.ever_act_val.sum())
        print(f"\n✅ global coverage:"
              f"  train dead {total-train_alive}/{total}"
              f" ({(total-train_alive)/total*100:.1f}%),"
              f" val dead {total-val_alive}/{total}"
              f" ({(total-val_alive)/total*100:.1f}%)")

    # ── inference helper ────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        h = self.esm(input_ids, attention_mask)
        return self.sae.encoder(h[attention_mask.bool()])

# ──────────────────────────────────────────────────────────────────────────────
# smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("running SAE smoke-test …")
    x = torch.randn(256, 1280)
    sae = SparseAutoencoder(1280, latent_dim=1024, topk=10)
    x_hat, acts = sae(x)
    assert x_hat.shape == x.shape
    logger.info("smoke-test ok — dead=%.1f%%", sae.dead_ratio(acts))

