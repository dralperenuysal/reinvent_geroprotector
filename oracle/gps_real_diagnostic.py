"""Paired diagnostic: for the same real LINCS compounds, compare GPS's
*predicted* SenMayo reversal score (from structure alone) against the
*real measured* SenMayo signal already computed in
data/processed/lincs_senmayo_screen_results.csv (Methods 2.2, real LINCS L1000
Level-5 data). This is the targeted diagnostic referenced in the manuscript's
"Methodological note" (Methods 2.2) and Discussion: it isolates whether GPS's
failure of the positive-control check (oracle/robustness_check.py) reflects a
GPS-specific bias or a property of the LINCS data itself.

GPS repo: https://github.com/Bin-Chen-Lab/GPS (Apache-2.0), cloned to gps/GPS/.
SenMayo gene list: Saul et al. 2022 Nat Commun Supplementary Data 1, downloaded
directly from Springer's ESM host (data/raw/senmayo/).

Usage: python oracle/gps_real_diagnostic.py
Output: data/processed/gps_real_diagnostic.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/gps/GPS/GPS4Drugs/code")

GPS_ROOT = f"{ROOT}/gps/GPS/GPS4Drugs/"
CELL_LINES = ["HEPG2_t0", "MCF7_t1", "PC3_t1", "VCAP_t1"]
SENMAYO_PATH = f"{ROOT}/data/raw/senmayo/senmayo_human_genes.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SAMPLE = 300
SEED = 42
POSITIVE_CONTROLS = ["dasatinib", "navitoclax", "panobinostat", "quercetin", "sirolimus", "vorinostat"]


def get_morgan_fp(smiles, radius=3, n_bits=1024):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
    bits = np.frombuffer(fp.ToBitString().encode(), dtype="u1").astype(np.float32) - ord("0")
    return bits


def catg_assign(probs, threshold=0.95):
    maxp = probs.max(axis=-1)
    argm = probs.argmax(axis=-1) - 1
    return np.where(maxp < threshold, 0, argm)


def main():
    t0 = time.time()

    # ---- real LINCS side: already-computed, real-measured per-compound results ----
    screen = pd.read_csv(f"{ROOT}/data/processed/lincs_senmayo_screen_results.csv")
    smiles_map = pd.read_pickle(f"{ROOT}/data/processed/lincs_smiles.pkl")
    df = screen.merge(smiles_map, on="pert_id", how="left")
    df = df[df.canonical_smiles.notna() & (df.canonical_smiles != "-666")]
    print(f"Real-data compounds with resolved SMILES: {len(df)}/{len(screen)}")

    rng = np.random.RandomState(SEED)
    pc_mask = df.pert_iname.str.lower().isin(POSITIVE_CONTROLS)
    pc_df = df[pc_mask].drop_duplicates(subset="pert_iname")
    print(f"Positive controls resolved: {len(pc_df)}/{len(POSITIVE_CONTROLS)} -> "
          f"{pc_df.pert_iname.str.lower().tolist()}")

    pool = df[~pc_mask]
    sample_df = pool.sample(n=min(N_SAMPLE, len(pool)), random_state=rng)
    work = pd.concat([pc_df, sample_df], ignore_index=True)
    work["is_positive_control"] = work.pert_iname.str.lower().isin(POSITIVE_CONTROLS)
    print(f"Scoring {len(work)} compounds with GPS "
          f"({work.is_positive_control.sum()} positive controls + {(~work.is_positive_control).sum()} library sample)...")

    # ---- GPS side: gene features, SenMayo gene overlap, trained models ----
    senmayo_genes = [l.strip() for l in open(SENMAYO_PATH)]
    gene_feat_full = pd.read_csv(
        GPS_ROOT + "data/input_gene_features/go_fingerprints_allGenesExt.csv", index_col=0
    )
    covered_genes = [g for g in senmayo_genes if g in gene_feat_full.index]
    print(f"SenMayo genes covered by GPS gene-feature space: {len(covered_genes)}/{len(senmayo_genes)}")
    gene_feat = gene_feat_full.loc[covered_genes].values.astype(np.float32)
    n_genes = gene_feat.shape[0]
    gene_feat_t = torch.from_numpy(gene_feat).to(DEVICE)

    models = {}
    for cl in CELL_LINES:
        m = torch.load(GPS_ROOT + f"code/results/{cl}/multi/model.pkl", map_location=DEVICE, weights_only=False)
        model0 = m["model0"].to(DEVICE)
        model0.eval()
        models[cl] = model0
    print(f"Loaded {len(models)} GPS cell-line models. Setup time: {time.time()-t0:.1f}s")

    # ---- batched inference: two scoring formulations, both reported ----
    # (1) discrete: argmax direction per gene at a 0.95 confidence threshold,
    #     then mean sign (matches oracle/gps_prototype_test.py exactly).
    # (2) continuous: mean(P(down) - P(up)) per gene, no thresholding -- this is
    #     the formulation that actually surfaces the systematic bias described
    #     in Methods 2.2, since the discrete score saturates into the "flat"
    #     bucket for most genes at the 0.95 threshold and washes the signal out.
    t_score = time.time()
    discrete_scores, continuous_scores = [], []
    with torch.no_grad():
        for _, row in work.iterrows():
            fp = get_morgan_fp(row.canonical_smiles)
            if fp is None:
                discrete_scores.append(np.nan)
                continuous_scores.append(np.nan)
                continue
            fp_t = torch.from_numpy(fp).to(DEVICE).unsqueeze(0).repeat(n_genes, 1)
            data = torch.cat([fp_t, gene_feat_t], dim=1)
            cl_probs = []
            for model0 in models.values():
                logits = model0(data)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                cl_probs.append(probs)
            median_probs = np.median(np.stack(cl_probs), axis=0)  # (n_genes, 3): [down, flat, up]
            directions = catg_assign(median_probs)
            discrete_scores.append(-directions.mean())
            continuous_scores.append((median_probs[:, 0] - median_probs[:, 2]).mean())

    work["gps_discrete_score"] = discrete_scores
    work["gps_continuous_score"] = continuous_scores
    print(f"Scoring wall time: {time.time()-t_score:.1f}s for {len(work)} compounds")

    out_cols = ["pert_id", "pert_iname", "is_positive_control", "canonical_smiles",
                "senmayo_mean_z", "reversal_score", "gps_discrete_score", "gps_continuous_score"]
    out = work[out_cols].dropna(subset=["gps_continuous_score"])
    out_path = f"{ROOT}/data/processed/gps_real_diagnostic.csv"
    out.to_csv(out_path, index=False)

    r_disc = np.corrcoef(out.reversal_score, out.gps_discrete_score)[0, 1]
    r_cont = np.corrcoef(out.reversal_score, out.gps_continuous_score)[0, 1]
    print(f"\nSaved {out_path} ({len(out)} compounds)")
    print(f"Pearson r(real reversal_score, GPS discrete score)   = {r_disc:.3f}")
    print(f"Pearson r(real reversal_score, GPS continuous score) = {r_cont:.3f}")
    print(f"GPS continuous score < 0 (predicts net SenMayo UP-regulation): "
          f"{(out.gps_continuous_score < 0).mean():.1%} of compounds")
    print(f"Real data: reversal_score < 0 (net up-regulation): "
          f"{(out.reversal_score < 0).mean():.1%} of compounds")
    print(f"\nPositive controls:")
    print(out[out.is_positive_control][["pert_iname", "reversal_score", "gps_discrete_score", "gps_continuous_score"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
