#!/usr/bin/env python
"""
Per-gene companion to lincs_senmayo_screen.py: instead of collapsing each
compound's SenMayo genes to one mean z-score (reversal_score), keep the
per-gene consensus z-score for every SenMayo gene. Reads only the SenMayo
gene rows (not the full ~10,174-gene BING background lincs_senmayo_screen.py
needs for its permutation null), so it is far cheaper.

Raw LINCS files (L5_GSE70138.gctx, L5_GSE92742.gctx, sig_info*.txt,
gene_info*.txt) are read directly from the aging_clock sibling project
(data/raw/lincs is gitignored/not present here, tens of GB); nothing is
copied, only referenced.

Output: data/processed/lincs_senmayo_pergene.csv, one row per compound
(pert_id, pert_iname, then one z-score column per SenMayo gene symbol).
"""
import os, warnings
import numpy as np, pandas as pd, h5py

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINCS = "/home/alperen/data/PycharmProjects/aging_clock/data/lincs"
SENMAYO_PATH = f"{ROOT}/data/raw/senmayo/senmayo_human_genes.txt"
OUT = f"{ROOT}/data/processed"

SOURCES = [dict(name="GSE70138", gctx=f"{LINCS}/L5_GSE70138.gctx",
                sig=f"{LINCS}/sig_info70.txt", gi=f"{LINCS}/gene_info70.txt"),
           dict(name="GSE92742", gctx=f"{LINCS}/L5_GSE92742.gctx",
                sig=f"{LINCS}/sig_info92.txt", gi=f"{LINCS}/gene_info92.txt")]


def as_str(a):
    a = np.asarray(a)
    if a.dtype.kind == "O":
        a = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])
    return np.char.strip(a.astype(str))


def main():
    senmayo_genes = [l.strip() for l in open(SENMAYO_PATH)]
    print(f">>> SenMayo genes (target): {len(senmayo_genes)}")

    frames = []
    gene_cols_ref = None
    for src in SOURCES:
        sig = pd.read_csv(src["sig"], sep="\t", low_memory=False)
        gi = pd.read_csv(src["gi"], sep="\t", low_memory=False)
        sym2id = dict(zip(gi.pr_gene_symbol.astype(str), gi.pr_gene_id.astype(str)))
        covered = [g for g in senmayo_genes if g in sym2id]
        gene_ids = [sym2id[g] for g in covered]

        with h5py.File(src["gctx"], "r") as f:
            row_ids = as_str(f["/0/META/ROW/id"][:])
            col_ids = as_str(f["/0/META/COL/id"][:])
            id2row = {r: i for i, r in enumerate(row_ids)}
            rows = np.array([id2row[g] for g in gene_ids if g in id2row])
            covered = [g for g, gid in zip(covered, gene_ids) if gid in id2row]
            print(f">>> {src['name']}: {len(covered)}/{len(senmayo_genes)} SenMayo genes "
                  f"found in this source's gene space")

            sig_idx = {s: i for i, s in enumerate(col_ids)}
            cp = sig[(sig.pert_type == "trt_cp") & sig.sig_id.isin(sig_idx)].copy()
            cols = np.array([sig_idx[s] for s in cp.sig_id])
            print(f">>> {src['name']}: {len(cp):,} compound signatures, "
                  f"{cp.pert_id.nunique():,} compounds")

            d = f["/0/DATA/0/matrix"]
            sig_axis0 = d.shape[0] == len(col_ids)
            n_tot = d.shape[0] if sig_axis0 else d.shape[1]

            r_ord = np.argsort(rows)
            rows_sorted, r_inv = rows[r_ord], np.argsort(r_ord)
            out = np.empty((len(cols), len(rows)), dtype=np.float32)
            col_out_pos = np.full(len(col_ids), -1, dtype=np.int64)
            col_out_pos[cols] = np.arange(len(cols))
            step = 5000
            for i in range(0, n_tot, step):
                j = min(i + step, n_tot)
                out_pos = col_out_pos[i:j]
                keep = out_pos >= 0
                if not keep.any():
                    print(f"    {j:,}/{n_tot:,}", end="\r", flush=True)
                    continue
                blk = d[i:j, :][:, rows_sorted] if sig_axis0 else d[rows_sorted, i:j].T
                blk = blk[:, r_inv]
                out[out_pos[keep]] = blk[keep]
                print(f"    {j:,}/{n_tot:,}", end="\r", flush=True)
            print()

        gdf = pd.DataFrame(out, columns=covered)
        gdf["pert_id"] = cp.pert_id.values
        gdf["pert_iname"] = cp.pert_iname.values
        gdf["source"] = src["name"]
        frames.append(gdf)
        if gene_cols_ref is None:
            gene_cols_ref = set(covered)
        else:
            gene_cols_ref &= set(covered)

    shared_genes = sorted(gene_cols_ref)
    print(f"\n>>> Genes present in both sources: {len(shared_genes)}/{len(senmayo_genes)}")

    all_sig = pd.concat([f[["pert_id", "pert_iname"] + shared_genes] for f in frames],
                         ignore_index=True)
    consensus = all_sig.groupby("pert_id", sort=False)[shared_genes].mean()
    names = all_sig.groupby("pert_id", sort=False)["pert_iname"].first()
    consensus.insert(0, "pert_iname", names)
    consensus = consensus.reset_index()

    out_path = f"{OUT}/lincs_senmayo_pergene.csv"
    consensus.to_csv(out_path, index=False)
    print(f">>> Saved {out_path} ({len(consensus):,} compounds x {len(shared_genes)} genes)")


if __name__ == "__main__":
    main()
