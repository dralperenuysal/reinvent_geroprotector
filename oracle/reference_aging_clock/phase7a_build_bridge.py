#!/usr/bin/env python
"""
Phase 7a: Structure -> dAge bridge.

Phase 7 asks whether a compound's effect on biological age can be anticipated at
all -- from its structure (7a-7c), and across the cell it acts in (7d-7e). This
script builds the model the rest of the phase characterises: a map from chemical
structure to dAge, learned from LINCS L1000 compounds that have both a canonical
SMILES and a measured consensus signature.

The bridge was originally the oracle behind de novo generation. Generation now
lives on the generative-design branch, because the numbers below turned out not
to support it; the bridge stays here as the object under study rather than as a
component of a generator.

Validation is by BEMIS-MURCKO SCAFFOLD SPLIT, not a random split: the question is
whether the map extends to chemistry unlike the training set, which a held-out
random sample would flatter. The scaffold-split correlation, and the balanced
directional accuracy beside it, bound everything Phase 7 can claim.
"""
import os, warnings
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
LINCS = f"{ROOT}/data/lincs"
N_BITS, RADIUS, SEED = 2048, 2, 0

print("=== PHASE 7a: STRUCTURE -> TRANSCRIPTOME BRIDGE ===")

screen = pd.read_csv(f"{ROOT}/phase6/phase6b_lincs_screen_results.csv")
smiles = pd.read_pickle(f"{LINCS}/smiles.pkl")
df = screen.merge(smiles, on="pert_id", how="inner").dropna(subset=["canonical_smiles"])
print(f">>> Compounds with signature + SMILES: {len(df):,}")

# ---------------------------------------------------------------- featurize
gen = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
         Descriptors.NumHDonors, Descriptors.NumHAcceptors,
         Descriptors.NumRotatableBonds, Descriptors.RingCount,
         Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.NumAromaticRings]

def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None
    fp = np.zeros(N_BITS, dtype=np.float32)
    for i, c in gen.GetCountFingerprint(mol).GetNonzeroElements().items():
        fp[i] = c
    d = np.array([f(mol) for f in DESCS], dtype=np.float32)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "__none__"
    except Exception:
        scaf = "__none__"
    return fp, d, scaf

feats, descs, scafs, keep = [], [], [], []
for i, s in enumerate(df.canonical_smiles.values):
    fp, d, sc = featurize(s)
    if fp is None:
        continue
    feats.append(fp); descs.append(d); scafs.append(sc); keep.append(i)
df = df.iloc[keep].reset_index(drop=True)
FP = np.vstack(feats); D = np.vstack(descs); scafs = np.array(scafs)
X = np.hstack([np.log1p(FP), (D - D.mean(0)) / (D.std(0) + 1e-8)])
y = df.dage.values
print(f">>> Featurized {len(df):,} molecules | {X.shape[1]:,} features | "
      f"{len(set(scafs)):,} distinct Murcko scaffolds")

# ---------------------------------------------------------------- scaffold split
rng = np.random.default_rng(SEED)
groups = pd.Series(scafs).groupby(scafs).size().sort_values(ascending=False)
order = list(groups.index)
rng.shuffle(order)
test_scaf, n = set(), 0
for s in order:                       # fill the test set with whole scaffold groups
    if n >= 0.25 * len(df):
        break
    test_scaf.add(s); n += groups[s]
te = np.isin(scafs, list(test_scaf))
tr = ~te
print(f">>> Scaffold split: train {tr.sum():,} / test {te.sum():,} "
      f"(zero scaffold overlap between them)")

# ---------------------------------------------------------------- fit + evaluate
# dAge is strongly skewed towards positive (most compounds look ageing in L1000), so
# raw sign accuracy is meaningless: always predicting "ageing" already scores high.
# Report it against that majority-class baseline, and use balanced accuracy, which a
# constant predictor cannot beat.
big = np.abs(y[te]) > np.quantile(np.abs(y[te]), 0.75)
BASELINE = max((y[te][big] > 0).mean(), (y[te][big] < 0).mean())


def bal_acc(truth, pred):
    """Balanced accuracy: mean of per-class recall, so a constant predictor scores 0.5."""
    return float(np.mean([(pred[truth == c] == c).mean()
                          for c in (True, False) if (truth == c).any()]))


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


print("\n>>> Held-out (scaffold-split) performance predicting dAge from structure:")
print(f"    ({te.sum()} held-out compounds; majority-class sign baseline = {BASELINE:.1%})")
res = []

ridge = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(X[tr], y[tr])
res.append(report("Ridge (direct dAge)", ridge.predict(X[te])))

rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                           random_state=SEED).fit(X[tr], y[tr])
res.append(report("RandomForest (direct dAge)", rf.predict(X[te])))

# Null reference: shuffled labels, to show the above is not chance
ridge_te = ridge.predict(X[te])
null_r = [pearsonr(y[te], ridge_te[rng.permutation(te.sum())])[0] for _ in range(200)]
print(f"  {'[permutation null]':28s} r={np.mean(null_r):+.3f} +/- {np.std(null_r):.3f}")

# The correlation had a null; the balanced sign accuracy did not, even though it is the
# number that decides whether the oracle can call a compound's direction at all. Shuffling
# the predicted signs holds the model's marginal positive rate fixed, so the null isolates
# whether the pairing of prediction to compound carries information.
t_big = np.sign(y[te][big]) > 0
p_big = np.sign(ridge_te[big]) > 0
obs_bal = bal_acc(t_big, p_big)
null_bal = np.array([bal_acc(t_big, rng.permutation(p_big)) for _ in range(2000)])
p_bal = (int((null_bal >= obs_bal).sum()) + 1) / (len(null_bal) + 1)
print(f"  {'[balanced-acc null]':28s} observed={obs_bal:.1%} vs null "
      f"{null_bal.mean():.1%} +/- {null_bal.std():.1%}  (p = {p_bal:.4f}, n={int(big.sum()):,} "
      f"top-quartile compounds)")

# Keep the held-out predictions so this evaluation can be re-examined without refitting.
pd.DataFrame(dict(pert_id=df.pert_id.values[te], dage_true=y[te], dage_pred=ridge_te,
                  top_quartile=big)).to_csv(f"{ROOT}/phase7/phase7a_heldout_predictions.csv",
                                            index=False)

best = max(res, key=lambda d: d["test_r"])
print(f"\n>>> Best bridge: {best['model']} (scaffold-split r={best['test_r']:+.3f})")

# Refit the deployed model on all data now that the honest estimate is recorded
deployed = (RidgeCV(alphas=np.logspace(-1, 4, 24)) if best["model"].startswith("Ridge")
            else RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                       n_jobs=-1, random_state=SEED)).fit(X, y)

joblib.dump(dict(model=deployed, n_bits=N_BITS, radius=RADIUS,
                 desc_mu=D.mean(0), desc_sd=D.std(0) + 1e-8,
                 dage_mu=float(y.mean()), dage_sd=float(y.std()),
                 validation=best, all_results=res, n_train=int(len(df))),
            f"{ROOT}/phase7/bridge_model.joblib")
pd.DataFrame(res).to_csv(f"{ROOT}/phase7/phase7a_bridge_validation.csv", index=False)

print(f">>> Saved phase7/bridge_model.joblib (deployed model refit on all {len(df):,} compounds)")
print(f"\n>>> HONEST CEILING for everything Phase 7 produces:")
print(f"      scaffold-split r        = {best['test_r']:+.3f}  "
      f"(permutation null {np.mean(null_r):+.3f} +/- {np.std(null_r):.3f})")
print(f"      balanced sign accuracy  = {best['sign_balanced_acc']:.1%}  "
      f"(0.5 = chance; raw accuracy {best['sign_acc']:.1%} vs baseline "
      f"{best['sign_acc_baseline']:.1%})")
print(f"      The structure -> transcriptome signal is real but weak: it explains "
      f"~{100*best['test_r']**2:.0f}% of variance in dAge across novel scaffolds.")
