#!/usr/bin/env python
"""
Plan B, step 2: structure -> SenMayo reversal-score bridge.

Adapted from aging_clock's phase7a_build_bridge.py. Learns a map from a
compound's structure (Morgan count fingerprint + physicochemical descriptors)
to its SenMayo reversal_score, as measured by lincs_senmayo_screen.py on real
LINCS L1000 data. This deployed model is the REINVENT4 oracle: during
generation REINVENT4 proposes SMILES that were never in LINCS, so scoring them
requires predicting from structure, not looking up a measured signature.

Validation is by Bemis-Murcko SCAFFOLD SPLIT, not a random split, for the same
reason as phase7a: the question is whether the map extends to chemistry unlike
the training set, which a held-out random sample would flatter.
"""
import os, sys, warnings
import numpy as np, pandas as pd, joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINCS = f"{ROOT}/data/raw/lincs"
OUT = f"{ROOT}/data/processed"
N_BITS, RADIUS, SEED = 2048, 2, 0
TARGET_COL = "reversal_score"

POSITIVE_CONTROL_NAMES = ["dasatinib", "navitoclax", "panobinostat", "quercetin", "sirolimus", "vorinostat"]


def load_positive_control_smiles():
    """Look up canonical SMILES from LINCS's own pert_info table instead of hand-typing
    structures from memory: a transcription error here (verified against this exact set --
    an initial hand-typed dasatinib SMILES was subtly wrong and flipped its predicted
    score from clearly reversing to clearly not) silently invalidates the whole robustness
    check, since the model would then be scored on the wrong molecule."""
    pert = pd.read_csv(f"{LINCS}/pert_info92.txt", sep="\t", low_memory=False)
    rows = pert[pert.pert_iname.str.lower().isin(POSITIVE_CONTROL_NAMES) &
                (pert.canonical_smiles != "-666")].drop_duplicates("pert_iname")
    found = dict(zip(rows.pert_iname.str.lower(), rows.canonical_smiles))
    missing = set(POSITIVE_CONTROL_NAMES) - set(found)
    if missing:
        raise SystemExit(f"Positive controls missing from pert_info92: {missing}")
    return found


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


def bal_acc(truth, pred):
    """Balanced accuracy: mean of per-class recall, so a constant predictor scores 0.5."""
    return float(np.mean([(pred[truth == c] == c).mean()
                          for c in (True, False) if (truth == c).any()]))


def main():
    print("=== PLAN B STEP 2: STRUCTURE -> SenMayo REVERSAL-SCORE BRIDGE ===")

    screen = pd.read_csv(f"{OUT}/lincs_senmayo_screen_results.csv")
    smiles = pd.read_pickle(f"{LINCS}/smiles.pkl")
    df = screen.merge(smiles, on="pert_id", how="inner")
    df = df[df.canonical_smiles != "-666"].dropna(subset=["canonical_smiles"])
    print(f">>> Compounds with screen score + SMILES: {len(df):,}")

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
    y = df[TARGET_COL].values
    print(f">>> Featurized {len(df):,} molecules | {X.shape[1]:,} features | "
          f"{len(set(scafs)):,} distinct Murcko scaffolds")

    # ---------------------------------------------------------------- scaffold split
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
    print(f">>> Scaffold split: train {tr.sum():,} / test {te.sum():,} "
          f"(zero scaffold overlap between them)")

    # reversal_score is roughly symmetric (it's a Z-score), so use the top-quartile-by-
    # magnitude majority-class baseline exactly as phase7a did for dAge.
    big = np.abs(y[te]) > np.quantile(np.abs(y[te]), 0.75)
    BASELINE = max((y[te][big] > 0).mean(), (y[te][big] < 0).mean())

    def report(tag, pred):
        r = pearsonr(y[te], pred)[0]
        rho = spearmanr(y[te], pred)[0]
        t, p = np.sign(y[te][big]) > 0, np.sign(pred[big]) > 0
        acc = (t == p).mean()
        bal = bal_acc(t, p)
        print(f"  {tag:28s} r={r:+.3f}  rho={rho:+.3f}  "
              f"sign-acc={acc:.1%} (baseline {BASELINE:.1%})  balanced={bal:.1%}")
        return dict(model=tag, test_r=r, test_rho=rho, sign_acc=acc,
                    sign_acc_baseline=float(BASELINE), sign_balanced_acc=bal)

    print("\n>>> Held-out (scaffold-split) performance predicting reversal_score from structure:")
    print(f"    ({te.sum()} held-out compounds; majority-class sign baseline = {BASELINE:.1%})")
    res = []

    ridge = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(X[tr], y[tr])
    res.append(report("Ridge (direct reversal_score)", ridge.predict(X[te])))

    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                               random_state=SEED).fit(X[tr], y[tr])
    res.append(report("RandomForest (direct reversal_score)", rf.predict(X[te])))

    ridge_te = ridge.predict(X[te])
    null_r = [pearsonr(y[te], ridge_te[rng.permutation(te.sum())])[0] for _ in range(200)]
    print(f"  {'[permutation null]':28s} r={np.mean(null_r):+.3f} +/- {np.std(null_r):.3f}")

    t_big = np.sign(y[te][big]) > 0
    p_big = np.sign(ridge_te[big]) > 0
    obs_bal = bal_acc(t_big, p_big)
    null_bal = np.array([bal_acc(t_big, rng.permutation(p_big)) for _ in range(2000)])
    p_bal = (int((null_bal >= obs_bal).sum()) + 1) / (len(null_bal) + 1)
    print(f"  {'[balanced-acc null]':28s} observed={obs_bal:.1%} vs null "
          f"{null_bal.mean():.1%} +/- {null_bal.std():.1%}  (p = {p_bal:.4f}, n={int(big.sum()):,} "
          f"top-quartile compounds)")

    pd.DataFrame(dict(pert_id=df.pert_id.values[te], score_true=y[te], score_pred=ridge_te,
                      top_quartile=big)).to_csv(f"{OUT}/lincs_bridge_heldout_predictions.csv",
                                                index=False)

    best = max(res, key=lambda d: d["test_r"])
    print(f"\n>>> Best bridge: {best['model']} (scaffold-split r={best['test_r']:+.3f})")

    deployed = (RidgeCV(alphas=np.logspace(-1, 4, 24)) if best["model"].startswith("Ridge")
                else RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                           n_jobs=-1, random_state=SEED)).fit(X, y)

    joblib.dump(dict(model=deployed, n_bits=N_BITS, radius=RADIUS,
                     desc_mu=desc_mu, desc_sd=desc_sd,
                     score_mu=float(y.mean()), score_sd=float(y.std()),
                     validation=best, all_results=res, n_train=int(len(df))),
                f"{OUT}/bridge_model.joblib")
    pd.DataFrame(res).to_csv(f"{OUT}/lincs_bridge_validation.csv", index=False)

    print(f">>> Saved data/processed/bridge_model.joblib (deployed model refit on all {len(df):,} compounds)")
    print(f"\n>>> HONEST CEILING for the Plan B bridge:")
    print(f"      scaffold-split r        = {best['test_r']:+.3f}  "
          f"(permutation null {np.mean(null_r):+.3f} +/- {np.std(null_r):.3f})")
    print(f"      balanced sign accuracy  = {best['sign_balanced_acc']:.1%}  "
          f"(0.5 = chance; raw accuracy {best['sign_acc']:.1%} vs baseline "
          f"{best['sign_acc_baseline']:.1%})")
    print(f"      Variance explained across novel scaffolds: ~{100*best['test_r']**2:.0f}%.")

    # ---------------------------------------------------------------- Section 3.0 robustness
    # Score the positive controls by STRUCTURE PREDICTION (the deployed model), not by
    # looking up their real measured LINCS signature -- this is what REINVENT4 actually
    # calls on generated molecules, most of which were never in LINCS.
    print("\n=== SECTION 3.0 ROBUSTNESS CHECK (bridge model's own structure-based predictions) ===")
    positive_controls = load_positive_control_smiles()
    pc_smi = list(positive_controls.values())
    pc_names = list(positive_controls.keys())
    pc_feats, pc_ok = [], []
    for s in pc_smi:
        fp, d, _ = featurize(s, gen, DESCS)
        if fp is None:
            pc_ok.append(False)
            continue
        pc_ok.append(True)
        pc_feats.append(np.concatenate([np.log1p(fp), (d - desc_mu) / desc_sd]))
    pc_X = np.vstack(pc_feats)
    pc_pred = deployed.predict(pc_X)

    lib_mean = y.mean()
    print(f"Library mean reversal_score (real measured data, all training compounds) = {lib_mean:+.4f}")
    above = 0
    j = 0
    for name, ok in zip(pc_names, pc_ok):
        if not ok:
            print(f"{name:<15}invalid SMILES")
            continue
        sc = pc_pred[j]; j += 1
        flag = " <-- above library mean" if sc > lib_mean else ""
        if sc > lib_mean:
            above += 1
        print(f"{name:<15}{sc:>+10.4f}{flag}")
    print(f"\n{above}/{sum(pc_ok)} positive controls score above the library mean (bridge model predictions).")


if __name__ == "__main__":
    main()
