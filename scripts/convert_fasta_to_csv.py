#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from Bio import SeqIO

MUT_REGEX = re.compile(r"([A-Z])(\d+)(?:→|->)([A-Z])")

def get_core_id(header: str) -> str:
    """Extract the stable UniProt ID (e.g. Q53XC0) from a header."""
    m = re.search(r"\|([A-Z0-9]+)\|", header)
    return m.group(1) if m else header.split()[0]

def extract_mutation(header: str):
    """Return (orig_aa, pos, new_aa) or None if no mutation tag."""
    m = MUT_REGEX.search(header)
    if not m:
        return None
    orig, pos, new = m.groups()
    return orig, int(pos) - 1, new  # convert to 0-based index

def main(fasta_path: str, output_csv: str):
    # load everything
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise SystemExit(f"no sequences found in {fasta_path}")

    # group by core ID
    by_id = defaultdict(list)
    for rec in records:
        pid = get_core_id(rec.description)
        by_id[pid].append(rec)

    rows = []
    for pid, recs in by_id.items():
        # find WT: first record lacking a mutation tag
        wts = [r for r in recs if extract_mutation(r.description) is None]
        if not wts:
            print(f"⚠ warning: no WT record found for {pid}, skipping group")
            continue
        wt_rec = wts[0]
        wt_seq = str(wt_rec.seq)

        # find all mutants
        muts = [r for r in recs if extract_mutation(r.description) is not None]
        if not muts:
            # no mutants for this protein
            continue

        for mut_rec in muts:
            tag = extract_mutation(mut_rec.description)
            assert tag, "logic error: expected mutation tag"
            orig, pos, new = tag

            # sanity check
            if wt_seq[pos] != orig:
                print(f"⚠ mismatch for {pid} at pos {pos+1}: "
                      f"WT has {wt_seq[pos]}, header says {orig}")

            mut_id = f"{pid}_{orig}{pos+1}{new}"
            rows.append([mut_id, wt_seq, str(mut_rec.seq)])

    # write out
    with open(output_csv, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["id", "wt_seq", "mut_seq"])
        w.writerows(rows)

    print(f"✅ wrote {len(rows)} mutant pairs for {len(by_id)} proteins → {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert a FASTA of WT+alanine-scan mutants to CSV id,wt_seq,mut_seq")
    p.add_argument("--fasta", required=True,
                   help="input FASTA (WT + mutants)")
    p.add_argument("--out",   required=True,
                   help="output CSV path")
    args = p.parse_args()
    main(args.fasta, args.out)

