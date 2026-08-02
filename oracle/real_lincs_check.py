"""Decisive check: using REAL measured LINCS L1000 z-scores (not GPS predictions),
do known senolytics/senomorphics show lower (more down-regulated) SenMayo gene
z-scores than a random library sample? This tests whether the "acute stress
response confound" seen in GPS's predictions is a GPS-specific artifact or a
real property of the LINCS transcriptomic data itself (which would affect
Plan B equally, since Plan B trains directly on this data).
"""
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

LINCS_DIR = "/home/alperen/projects/reinvent_geroprotector/data/raw/lincs/"
GCTX = LINCS_DIR + "L5_GSE92742.gctx"
POSITIVE_CONTROLS = ["dasatinib", "navitoclax", "panobinostat", "quercetin", "sirolimus", "vorinostat"]
N_LIBRARY_SAMPLE = 300
SEED = 42


def main():
    sig_info = pd.read_csv(LINCS_DIR + "sig_info92.txt", sep="\t", low_memory=False)
    gene_info = pd.read_csv(LINCS_DIR + "gene_info92.txt", sep="\t")
    senmayo = [l.strip() for l in open("/home/alperen/projects/reinvent_geroprotector/data/raw/senmayo/senmayo_human_genes.txt")]

    gene_info_idx = gene_info.set_index("pr_gene_symbol")
    senmayo_covered = [g for g in senmayo if g in gene_info_idx.index]
    senmayo_gene_ids = gene_info_idx.loc[senmayo_covered, "pr_gene_id"].astype(str).tolist()
    print(f"SenMayo genes covered by LINCS L1000 (measured/inferred): {len(senmayo_covered)}/{len(senmayo)}")

    trt_cp = sig_info[sig_info["pert_type"] == "trt_cp"]

    rng = np.random.RandomState(SEED)
    pc_sig_ids = {}
    for name in POSITIVE_CONTROLS:
        rows = trt_cp[trt_cp["pert_iname"].str.lower() == name]
        if len(rows) == 0:
            continue
        take = rows["sig_id"].sample(n=min(20, len(rows)), random_state=rng).tolist()
        pc_sig_ids[name] = take

    other = trt_cp[~trt_cp["pert_iname"].str.lower().isin(POSITIVE_CONTROLS)]
    lib_compounds = other["pert_iname"].dropna().unique()
    lib_sample_compounds = rng.choice(lib_compounds, size=min(N_LIBRARY_SAMPLE, len(lib_compounds)), replace=False)
    lib_sig_ids = []
    for c in lib_sample_compounds:
        rows = other[other["pert_iname"] == c]
        lib_sig_ids.append(rows["sig_id"].sample(n=1, random_state=rng).iloc[0])

    all_sig_ids = [sid for v in pc_sig_ids.values() for sid in v] + lib_sig_ids
    print(f"Reading {len(all_sig_ids)} signatures x {len(senmayo_gene_ids)} SenMayo genes from GCTX...")

    gct = parse(GCTX, rid=senmayo_gene_ids, cid=all_sig_ids)
    data = gct.data_df  # genes x sig_ids

    pc_means = {}
    for name, sids in pc_sig_ids.items():
        present = [s for s in sids if s in data.columns]
        pc_means[name] = data[present].values.mean()

    lib_present = [s for s in lib_sig_ids if s in data.columns]
    lib_vals = data[lib_present].values.mean(axis=0)  # mean SenMayo z-score per signature
    lib_mean = lib_vals.mean()

    print(f"\nLibrary sample: n={len(lib_present)}, mean SenMayo z-score={lib_mean:+.4f}, std={lib_vals.std():.4f}")
    print(f"Library percentiles: p10={np.percentile(lib_vals,10):+.4f} p50={np.percentile(lib_vals,50):+.4f} p90={np.percentile(lib_vals,90):+.4f}")

    print("\nPositive controls (mean measured SenMayo z-score across their signatures):")
    below = 0
    for name, m in pc_means.items():
        pct = (lib_vals < m).mean() * 100
        flag = " <-- below library mean (down-regulating SenMayo, GOOD)" if m < lib_mean else ""
        if m < lib_mean:
            below += 1
        print(f"{name:<15}{m:>+10.4f}{pct:>10.1f}%ile{flag}")

    print(f"\n{below}/{len(pc_means)} positive controls show LOWER (more down-regulated) mean SenMayo z-score than the library mean.")
    print("(Lower/negative = down-regulation = desired reversal direction)")


if __name__ == "__main__":
    main()
