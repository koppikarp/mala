#!/usr/bin/env python3
"""
latent_heatmap.py

generate per-residue Δactivations for selected latents.

usage:
  python latent_heatmap.py \
      --ckpt outputs/sae/checkpoints/stepstep=0100000.ckpt \
      --cfg  config/esm2_l24.yaml \
      --pairs mutants.csv \
      --latents 2874 1700 3353 \
      --out    heatmaps.npz
"""
import argparse, csv, json, yaml
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from scripts.sae_components import SAELightningModule

# ------------- helpers -------------------------------------------------
def load_model(ckpt, cfg, device):
    model = SAELightningModule.load_from_checkpoint(ckpt, **cfg)
    model.eval().to(device)
    tok   = AutoTokenizer.from_pretrained(cfg["model_name"])
    return model, tok

def activations(seq, model, tok, device):
    toks = tok(seq, add_special_tokens=False, return_tensors="pt")
    ids  = toks["input_ids"].to(device)
    mask = (ids != tok.pad_token_id).long()
    with torch.no_grad():
        h = model.esm(ids, mask)           # [1,L,d]
        _, a = model.sae(h.squeeze(0))     # [L,latent]
    return a.cpu()

# ------------- main ----------------------------------------------------
def main(a):
    cfg = yaml.safe_load(open(a.cfg))
    model, tok = load_model(a.ckpt, cfg, a.device)
    lat_idx = [int(x) for x in a.latents]   # list of ints

    heatmaps = {}   # id -> (len_seq, n_latents) array
    with open(a.pairs) as f:
        rdr = csv.DictReader(f)
        for row in tqdm(rdr, desc="heat-maps"):
            wt  = activations(row["wt_seq"],  model, tok, a.device)
            mut = activations(row["mut_seq"], model, tok, a.device)
            delta = (mut - wt).abs()[:, lat_idx]   # |Δ| for chosen latents
            heatmaps[row["id"]] = delta.numpy()

    np.savez_compressed(a.out, **heatmaps)
    print(f"✅ wrote {len(heatmaps)} heat-maps to {a.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg",  required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--latents", nargs="+", required=True,
                   help="latent IDs to track")
    p.add_argument("--out", default="heatmaps.npz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    main(args)

