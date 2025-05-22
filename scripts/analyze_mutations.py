#!/usr/bin/env python3
import argparse, csv, yaml, json, math
from pathlib import Path
from collections import Counter
from itertools import islice

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from scripts.sae_components import SAELightningModule

# ───────────────────────────────────────── helpers
def load_model(ckpt, cfg, device):
    model = SAELightningModule.load_from_checkpoint(ckpt, **cfg)
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    return model, tokenizer

def forward_batch(model, ids, mask):
    """ESM→SAE forward for a padded batch."""
    with torch.no_grad():
        h = model.esm(ids, mask)        # [B,L,d]
        B, L, _ = h.shape
        acts = []
        for b in range(B):
            # flatten real tokens of this seq only
            real = mask[b].bool()
            _, a = model.sae(h[b, real])  # [Li, latent]
            acts.append(a.cpu())
    return acts                            # list of [Li, latent]

def chunk_forward(model, tokenizer, seq, device, chunk_len):
    """fallback for very long sequences; processes one seq in windows."""
    ids = tokenizer(seq, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    mask = (ids != tokenizer.pad_token_id).long()
    acts = []
    for s in range(0, ids.size(1), chunk_len):
        e = min(s + chunk_len, ids.size(1))
        acts.extend(forward_batch(model, ids[:, s:e], mask[:, s:e])[0])
    return torch.stack(acts)

# ───────────────────────────────────────── main
def batched(iterable, n):
    """yield lists of length ≤ n."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def main(args):
    cfg = yaml.safe_load(open(args.cfg))
    model, tokenizer = load_model(args.ckpt, cfg, args.device)

    latent_hist = Counter()
    total_rows  = sum(1 for _ in open(args.pairs)) - 1  # minus header

    with open(args.pairs) as f_pairs,\
         open(args.out, "w", newline="") as f_out:

        reader  = csv.DictReader(f_pairs)
        writer  = csv.writer(f_out)
        writer.writerow(["id", "L1_mean", "L2_mean", "top_latents"])

        for batch_rows in tqdm(batched(reader, args.batch_size),
                               total=math.ceil(total_rows/args.batch_size),
                               desc="analysing"):

            seqs = [r["wt_seq"]  for r in batch_rows] + \
                   [r["mut_seq"] for r in batch_rows]

            # tokenise & pad
            toks = tokenizer(seqs, add_special_tokens=False,
                             padding=True, return_tensors="pt")
            ids  = toks["input_ids"].to(args.device)
            mask = (ids != tokenizer.pad_token_id).long()

            # quick path if under chunk_len
            if ids.size(1) <= args.chunk_len:
                acts = forward_batch(model, ids, mask)
            else:
                # fallback to per-seq chunking
                acts = [chunk_forward(model, tokenizer, s, args.device, args.chunk_len)
                        for s in seqs]

            # split back into wt/mut pairs
            for row, a_wt, a_mut in zip(batch_rows, acts[:len(batch_rows)], acts[len(batch_rows):]):
                delta = a_mut - a_wt
                l1    = delta.abs().sum(1)
                l2    = delta.pow(2).sum(1).sqrt()

                latent_shift = delta.abs().max(0).values
                top_idx = torch.topk(latent_shift, args.topk).indices.tolist()
                latent_hist.update(top_idx)

                writer.writerow([row["id"],
                                 f"{l1.mean():.4f}",
                                 f"{l2.mean():.4f}",
                                 json.dumps(top_idx)])

    # save histogram
    hist_path = Path(args.out).with_suffix(".latent_hist.csv")
    with open(hist_path, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["latent_id", "count"])
        for idx, cnt in latent_hist.most_common():
            w.writerow([idx, cnt])

    print(f"✅ finished → {args.out} & {hist_path}")

# ───────────────────────────────────────── CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg",  required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", default="delta_summary.csv")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--chunk_len",  type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    main(args)

