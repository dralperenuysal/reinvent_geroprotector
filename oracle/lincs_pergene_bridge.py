#!/usr/bin/env python
"""
Per-gene structure -> SenMayo-gene-z-score bridge models, for a mechanistic
gene-level view of the top-10 Run-2 candidates (Results 3.3 / Discussion).

The deployed oracle (lincs_structure_bridge.py) predicts one aggregate
reversal_score; it has no gene-level resolution. This script trains a
separate, lightweight Ridge model per SenMayo gene (same Morgan fingerprint +
physicochemical featurization as the deployed oracle, same scaffold-split
discipline) on the real per-gene z-scores extracted by
lincs_senmayo_pergene.py, restricted to the 24 genes most correlated with the
aggregate reversal_score (dominated by canonical SASP cytokines/chemokines/
MMPs: IL6, IL1B, MMP1, MMP12, CCL2/4/7/8, CXCL2/3, etc.) -- not GPS.

Output: data/processed/pergene_bridge_predictions.csv (top-10 candidates x
24 genes, predicted z-scores) and data/processed/pergene_bridge_validation.csv
(per-gene scaffold-split r).
"""
import os, warnings
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"
N_BITS, RADIUS, SEED, N_GENES = 2048, 2, 0, 24


def featurize(smi, gen, descs):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None
    fp = np.zeros(N_BITS, dtype=np.float32)
    for i, c in gen.GetCountFingerprint(mol).GetNonzeroElements().items():
        fp[i] = c
    d = np.array([f(mol) for f in descs], dtype=np.float32)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "__none__"
    except Exception:
        scaf = "__none__"
    return fp, d, scaf


def main():
    pergene = pd.read_csv(f"{PROC}/lincs_senmayo_pergene.csv")
    screen = pd.read_csv(f"{PROC}/lincs_senmayo_screen_results.csv")[["pert_id", "reversal_score"]]
    smiles_map = pd.read_pickle(f"{PROC}/lincs_smiles.pkl")

    gene_cols = [c for c in pergene.columns if c not in ("pert_id", "pert_iname")]
    merged = pergene.merge(screen, on="pert_id", how="inner")
    corrs = merged[gene_cols].corrwith(merged["reversal_score"]).abs().sort_values(ascending=False)
    genes = list(corrs.head(N_GENES).index)
    print(f">>> Top {N_GENES} SenMayo genes by |corr| with reversal_score:")
    print(", ".join(genes))

    df = pergene.merge(smiles_map, on="pert_id", how="left")
    df = df[df.canonical_smiles.notna() & (df.canonical_smiles != "-666")].reset_index(drop=True)
    print(f">>> Compounds with per-gene data + SMILES: {len(df):,}")

    gen = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
             Descriptors.NumHDonors, Descriptors.NumHAcceptors,
             Descriptors.NumRotatableBonds, Descriptors.RingCount,
             Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.NumAromaticRings]

    feats, descs, scafs, keep = [], [], [], []
    for i, s in enumerate(df.canonical_smiles.values):
        fp, d, sc = featurize(s, gen, DESCS)
        if fp is None:
            continue
        feats.append(fp); descs.append(d); scafs.append(sc); keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    FP = np.vstack(feats); D = np.vstack(descs); scafs = np.array(scafs)
    desc_mu, desc_sd = D.mean(0), D.std(0) + 1e-8
    X = np.hstack([np.log1p(FP), (D - desc_mu) / desc_sd])
    print(f">>> Featurized {len(df):,} molecules | {X.shape[1]:,} features | "
          f"{len(set(scafs)):,} scaffolds")

    # one scaffold split, shared across all 24 per-gene models (same discipline as the
    # main bridge model, and cheaper than a fresh split per gene)
    rng = np.random.default_rng(SEED)
    groups = pd.Series(scafs).groupby(scafs).size().sort_values(ascending=False)
    order = list(groups.index)
    rng.shuffle(order)
    test_scaf, n = set(), 0
    for s in order:
        if n >= 0.25 * len(df):
            break
        test_scaf.add(s); n += groups[s]
    te = np.isin(scafs, list(test_scaf))
    tr = ~te
    print(f">>> Scaffold split: train {tr.sum():,} / test {te.sum():,}")

    val_rows = []
    models = {}
    for g in genes:
        y = df[g].values
        ridge = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(X[tr], y[tr])
        r = pearsonr(y[te], ridge.predict(X[te]))[0]
        val_rows.append(dict(gene=g, test_r=r, n=len(df)))
        # refit on all data for the deployed per-gene predictor
        models[g] = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(X, y)

    val = pd.DataFrame(val_rows).sort_values("test_r", ascending=False)
    val.to_csv(f"{PROC}/pergene_bridge_validation.csv", index=False)
    print(f"\n>>> Per-gene scaffold-split r: mean={val.test_r.mean():+.3f}, "
          f"median={val.test_r.median():+.3f}, range=[{val.test_r.min():+.3f}, {val.test_r.max():+.3f}]")
    print(val.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- predict for the top-10 Run-2 candidates ----
    top10 = pd.read_csv(f"{PROC}/run2/top_candidates_diverse.csv")
    pred_rows = []
    for _, row in top10.iterrows():
        fp, d, _ = featurize(row.canonical, gen, DESCS)
        if fp is None:
            continue
        x = np.concatenate([np.log1p(fp), (d - desc_mu) / desc_sd]).reshape(1, -1)
        pred_rows.append({"canonical": row.canonical,
                           **{g: float(models[g].predict(x)[0]) for g in genes}})
    pred = pd.DataFrame(pred_rows)
    pred.insert(0, "rank", range(1, len(pred) + 1))
    pred.to_csv(f"{PROC}/pergene_bridge_predictions.csv", index=False)
    print(f"\n>>> Saved {PROC}/pergene_bridge_predictions.csv ({len(pred)} candidates x {len(genes)} genes)")
    print(f">>> Saved {PROC}/pergene_bridge_validation.csv")


if __name__ == "__main__":
    main()
