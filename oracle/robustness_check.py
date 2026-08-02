"""Section 3.0 robustness check: do known senolytics/senomorphics score above
the LINCS library average on the GPS-based SenMayo reversal oracle?
"""
import sys
import time

sys.path.insert(0, "/home/alperen/projects/reinvent_geroprotector/oracle")
from gps_oracle import score_compounds  # noqa: E402

import numpy as np
import pandas as pd

LINCS_DIR = "/home/alperen/projects/reinvent_geroprotector/data/raw/lincs/"
POSITIVE_CONTROLS = ["dasatinib", "navitoclax", "panobinostat", "quercetin", "sirolimus", "vorinostat"]
N_LIBRARY_SAMPLE = 1000
SEED = 42


def main():
    t0 = time.time()
    df = pd.read_csv(LINCS_DIR + "pert_info92.txt", sep="\t", low_memory=False)
    df = df[(df["pert_type"] == "trt_cp") & (df["canonical_smiles"] != "-666")].drop_duplicates(
        subset="pert_iname"
    )
    print(f"Compounds with valid SMILES in GSE92742 pert_info: {len(df)}")

    pc_df = df[df["pert_iname"].str.lower().isin(POSITIVE_CONTROLS)][["pert_iname", "canonical_smiles"]]
    print(f"Positive controls found: {len(pc_df)}/{len(POSITIVE_CONTROLS)}")

    rng = np.random.RandomState(SEED)
    library_pool = df[~df["pert_iname"].str.lower().isin(POSITIVE_CONTROLS)]
    sample_df = library_pool.sample(n=min(N_LIBRARY_SAMPLE, len(library_pool)), random_state=rng)

    print(f"Scoring {len(pc_df)} positive controls + {len(sample_df)} library sample compounds with GPS...")
    t_score_start = time.time()

    pc_scores = score_compounds(pc_df["canonical_smiles"].tolist())
    lib_scores, lib_valid = score_compounds(sample_df["canonical_smiles"].tolist(), return_valid_mask=True)

    t_score_end = time.time()
    print(f"Scoring wall time: {t_score_end - t_score_start:.2f}s")

    lib_valid_scores = lib_scores[lib_valid]
    lib_mean = np.nanmean(lib_valid_scores)
    lib_std = np.nanstd(lib_valid_scores)

    print(f"\nLibrary sample: n={len(lib_valid_scores)} (valid SMILES), mean={lib_mean:+.4f}, std={lib_std:.4f}")
    print(f"Library percentiles: p10={np.percentile(lib_valid_scores,10):+.4f} "
          f"p50={np.percentile(lib_valid_scores,50):+.4f} p90={np.percentile(lib_valid_scores,90):+.4f}")

    print("\nPositive controls:")
    print(f"{'compound':<15}{'score':>10}{'percentile_in_library':>25}")
    above_avg = 0
    for name, sc in zip(pc_df["pert_iname"], pc_scores):
        if np.isnan(sc):
            print(f"{name:<15}{'invalid SMILES':>10}")
            continue
        pct = (lib_valid_scores < sc).mean() * 100
        flag = " <-- above library mean" if sc > lib_mean else ""
        if sc > lib_mean:
            above_avg += 1
        print(f"{name:<15}{sc:>+10.4f}{pct:>24.1f}%{flag}")

    n_valid_pc = pc_df.shape[0]
    print(f"\n{above_avg}/{n_valid_pc} positive controls score above the library mean ({lib_mean:+.4f}).")
    print(f"Total elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
