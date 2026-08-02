"""Prototype: fast, vectorized GPS inference for a small set of SenMayo genes.
Bypasses GPS4Drugs' legacy per-gene Python2.7 subprocess pipeline entirely.
"""
import sys
import time

sys.path.insert(0, "/home/alperen/projects/reinvent_geroprotector/gps/GPS/GPS4Drugs/code")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

GPS_ROOT = "/home/alperen/projects/reinvent_geroprotector/gps/GPS/GPS4Drugs/"
CELL_LINES = ["HEPG2_t0", "MCF7_t1", "PC3_t1", "VCAP_t1"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENMAYO_PATH = "/home/alperen/projects/reinvent_geroprotector/data/raw/senmayo/senmayo_human_genes.txt"


def get_morgan_fp(smiles, radius=3, n_bits=1024):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
    bits = np.frombuffer(fp.ToBitString().encode(), dtype="u1").astype(np.float32) - ord("0")
    return bits


def catg_assign(probs, threshold=0.95):
    # probs: (..., 3) softmax output. -1=down, 0=no-change, +1=up
    maxp = probs.max(axis=-1)
    argm = probs.argmax(axis=-1) - 1
    return np.where(maxp < threshold, 0, argm)


def main():
    t0 = time.time()

    senmayo_genes = [l.strip() for l in open(SENMAYO_PATH)]

    gene_feat_full = pd.read_csv(
        GPS_ROOT + "data/input_gene_features/go_fingerprints_allGenesExt.csv", index_col=0
    )
    covered_genes = [g for g in senmayo_genes if g in gene_feat_full.index]
    print(f"SenMayo genes covered by GPS gene-feature space: {len(covered_genes)}/{len(senmayo_genes)}")
    gene_feat = gene_feat_full.loc[covered_genes].values.astype(np.float32)  # (n_genes, 1107)
    n_genes = gene_feat.shape[0]

    t_load_genes = time.time()
    print(f"Gene feature load: {t_load_genes - t0:.2f}s")

    models = {}
    for cl in CELL_LINES:
        m = torch.load(GPS_ROOT + f"code/results/{cl}/multi/model.pkl", map_location=DEVICE, weights_only=False)
        model0 = m["model0"].to(DEVICE)
        model0.eval()
        models[cl] = model0
    t_load_models = time.time()
    print(f"Model load ({len(CELL_LINES)} cell lines): {t_load_models - t_load_genes:.2f}s")

    test_smiles = {
        "quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    }

    gene_feat_t = torch.from_numpy(gene_feat).to(DEVICE)

    for name, smi in test_smiles.items():
        t_start = time.time()
        drug_fp = get_morgan_fp(smi)
        if drug_fp is None:
            print(f"{name}: invalid SMILES")
            continue
        drug_fp_t = torch.from_numpy(drug_fp).to(DEVICE).unsqueeze(0).repeat(n_genes, 1)  # (n_genes, 1024)
        data = torch.cat([drug_fp_t, gene_feat_t], dim=1)  # (n_genes, 2131)

        cl_probs = []
        with torch.no_grad():
            for cl, model0 in models.items():
                logits = model0(data)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                cl_probs.append(probs)
        median_probs = np.median(np.stack(cl_probs), axis=0)  # (n_genes, 3)
        directions = catg_assign(median_probs)  # -1/0/+1 per gene

        reversal_score = -directions.mean()  # sign-flipped: down-regulating SenMayo = positive score

        elapsed = time.time() - t_start
        n_down = (directions == -1).sum()
        n_up = (directions == 1).sum()
        n_flat = (directions == 0).sum()
        print(
            f"{name}: reversal_score={reversal_score:+.4f} "
            f"(down={n_down}, up={n_up}, flat={n_flat}) time={elapsed:.3f}s"
        )

    print(f"Total wall time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
