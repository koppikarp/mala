#!/usr/bin/env python3
"""
analyse_delta_summary.py

usage:
  python analyse_delta_summary.py --csv delta_summary.csv --topn 20
produces:
  - mutation_ranking.csv  (sorted by L2_mean desc)
  - latent_histogram.csv  (latents ranked by popularity)
  - prints a quick text summary
"""
import csv, argparse, json
from collections import Counter

def main(args):
    muts = []
    latent_hits = Counter()

    with open(args.csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            row["L1_mean"] = float(row["L1_mean"])
            row["L2_mean"] = float(row["L2_mean"])
            top = json.loads(row["top_latents"])
            row["top_latents"] = top
            latent_hits.update(top)
            muts.append(row)

    # ── write mutation ranking ───────────────────────────────────────────
    muts.sort(key=lambda r: r["L2_mean"], reverse=True)
    with open("mutation_ranking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank","id","L1_mean","L2_mean","top_latents"])
        for i, m in enumerate(muts, 1):
            w.writerow([i, m["id"], f"{m['L1_mean']:.4f}",
                        f"{m['L2_mean']:.4f}", json.dumps(m["top_latents"])])

    # ── write latent histogram ───────────────────────────────────────────
    with open("latent_histogram.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latent_id","count"])
        for lat, cnt in latent_hits.most_common():
            w.writerow([lat, cnt])

    # ── quick console summary ────────────────────────────────────────────
    print(f"total mutants analysed  : {len(muts)}")
    print(f"unique latents activated: {len(latent_hits)} / 4096")
    print(f"top {args.topn} most-responsive latents:")
    for lat, cnt in latent_hits.most_common(args.topn):
        print(f"  latent {lat:<4}  hits {cnt}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="delta_summary.csv produced by analyse_mutations.py")
    p.add_argument("--topn", type=int, default=20,
                   help="how many latents to print in console")
    args = p.parse_args()
    main(args)

