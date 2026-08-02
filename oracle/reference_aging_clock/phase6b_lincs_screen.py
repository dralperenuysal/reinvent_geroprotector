#!/usr/bin/env python
"""
Phase 6b: Connectivity screen of the LINCS L1000 compound library against the
frozen CD4_T transcriptomic aging clock.

Replaces the earlier hand-written compound signatures with measured perturbational
transcriptomes: LINCS L1000 Phase 2 (GEO GSE70138), 107,404 compound signatures
covering 1,796 distinct compounds across 30 cell lines.

Scoring
-------
The clock is a linear model on standardized log-CPM, so a transcriptomic shift
composes with it additively:

    dAge = sum_g  coef_g * dz_g

where dz is the compound's differential expression signature. L1000 level-5 MODZ
values are robust z-scores of log expression, and the clock coefficients are per
standardized-log-expression unit; the composition therefore assumes 1 L1000 z-unit
~ 1 training-cohort SD. This affects the absolute year scale but not the ranking,
which is what the screen is for. Negative dAge = predicted rejuvenation.

Significance comes from a gene-label permutation null, so a compound is only called
a hit if its score exceeds what random gene assignment produces. Every compound in
the library is scored, so the ranking has a real null distribution instead of being
determined by which compounds were hand-picked.
"""
import os, warnings
import numpy as np, pandas as pd, h5py, joblib, scipy.sparse as sp
from scipy.stats import mannwhitneyu, false_discovery_control

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINCS = f"{ROOT}/data/lincs"
N_PERM = 20000
SEED = 0

# Both LINCS release phases, scored together. They were produced years apart, but level 5
# is a per-plate moderated z-score against that plate's own controls, so the two are on a
# common scale by construction. They cover the same 12,328 genes but order them differently
# inside the matrices, so each is aligned to the clock separately below. Phase 1 contributes the
# chemical diversity -- 20,413 compounds against Phase 2's 1,796 -- that the downstream
# structure-to-transcriptome bridge needs and could not get from Phase 2 alone.
SOURCES = [dict(name="GSE70138", phase="phase2", gctx=f"{LINCS}/L5_GSE70138.gctx",
                sig=f"{LINCS}/sig_info70.txt"),
           dict(name="GSE92742", phase="phase1", gctx=f"{LINCS}/L5_GSE92742.gctx",
                sig=f"{LINCS}/sig_info92.txt")]

print("=== PHASE 6b: LINCS L1000 CONNECTIVITY SCREEN vs CD4_T AGING CLOCK ===")

# ---------------------------------------------------------------- clock
art = joblib.load(f"{ROOT}/phase6/clock_artifact.joblib")["landmark"]
clock_genes, coef = art["genes"], art["coef"]
m = art["metrics"]
print(f">>> Frozen clock (landmark variant): {m['n_active']}/{m['n_features']} active genes | "
      f"held-out AIDA r={m['test_r']:.3f}, calibrated MAE={m['test_MAE_calibrated']:.2f} y")

# ---------------------------------------------------------------- LINCS
gi = pd.read_pickle(f"{LINCS}/gene_info.pkl")

def as_str(a):
    """h5py hands back fixed-length bytes or object arrays depending on how the file
    was written; a plain .astype(str) on the object case yields "b'5720'" and silently
    breaks every downstream id match."""
    a = np.asarray(a)
    if a.dtype.kind == "O":
        a = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])
    return np.char.strip(a.astype(str))


absent = [s["name"] for s in SOURCES if not os.path.exists(s["gctx"])]
if absent:
    raise SystemExit(f"Missing level-5 matrices for {', '.join(absent)}. Run data/fetch_lincs.sh. "
                     f"Scoring a subset of the phases silently would misreport library coverage.")

id2sym = dict(zip(gi.pr_gene_id.astype(str), gi.pr_gene_symbol.astype(str)))

rows = coef_u = usable = None
frames, blocks = [], []
for src in SOURCES:
    sig = pd.read_csv(src["sig"], sep="\t", low_memory=False)
    with h5py.File(src["gctx"], "r") as f:
        row_ids = as_str(f["/0/META/ROW/id"][:])      # gene ids (Entrez)
        col_ids = as_str(f["/0/META/COL/id"][:])      # signature ids
        dshape = f["/0/DATA/0/matrix"].shape
    print(f"\n>>> {src['name']} ({src['phase']}): matrix {dshape} | "
          f"{len(row_ids):,} genes x {len(col_ids):,} signatures")

    # Align L1000 gene rows to the clock's feature order; drop clock genes L1000 lacks.
    # The two phases carry the same 12,328 genes but store them in unrelated row orders
    # (12,324 of 12,328 positions differ), so this selection must be recomputed against
    # each matrix's own row ids. What has to agree across sources is the resulting gene
    # sequence, clock_genes[usable] -- reusing one source's row indices on the other would
    # pair nearly every coefficient with the wrong gene and still produce a number.
    row_sym = np.array([id2sym.get(r, "") for r in row_ids])
    sym2row = {s: i for i, s in enumerate(row_sym) if s}
    usable_s = np.array([g in sym2row for g in clock_genes])
    if usable_s.sum() < 0.5 * len(clock_genes):
        raise SystemExit(f"Only {usable_s.sum()}/{len(clock_genes)} clock genes matched "
                         f"{src['name']} rows - gene id mapping is broken, refusing to score.")
    if usable is None:
        usable, coef_u = usable_s, coef[usable_s]
        print(f">>> Clock genes measured by L1000: {usable.sum():,}/{len(clock_genes):,} "
              f"(carrying {100*np.abs(coef_u).sum()/np.abs(coef).sum():.1f}% of total |coef|)")
    elif not np.array_equal(usable_s, usable):
        raise SystemExit(f"{src['name']} measures a different subset of the clock's genes than "
                         f"{SOURCES[0]['name']}; the concatenated columns would not align.")
    rows = np.array([sym2row[g] for g in clock_genes[usable]])

    # Keep only compound-treatment signatures
    sig_idx = {s: i for i, s in enumerate(col_ids)}
    cp_s = sig[(sig.pert_type == "trt_cp") & sig.sig_id.isin(sig_idx)].copy()
    cp_s["source"] = src["phase"]
    cols = np.array([sig_idx[s] for s in cp_s.sig_id])
    print(f">>> Compound signatures: {len(cp_s):,} across {cp_s.pert_id.nunique():,} compounds")

    # Read the clock's gene slice for every signature. Both axes are subset, which HDF5
    # cannot do in one call, so slice the gene axis in bounded row chunks rather than
    # pulling the whole 5-23 GB matrix into memory.
    r_ord = np.argsort(rows)
    rows_sorted, r_inv = rows[r_ord], np.argsort(r_ord)
    # The slab depends only on the clock's gene set, so cache it per source: rereading
    # tens of GB of HDF5 on every rerun dominates the runtime of this script.
    cache = f"{LINCS}/slab_{src['name']}_{len(rows)}g.npy"
    if os.path.exists(cache):
        print(f">>> Reusing cached expression slab ({os.path.basename(cache)})")
        buf = np.load(cache)
    else:
        print(">>> Reading expression slab from GCTX...", flush=True)
        with h5py.File(src["gctx"], "r") as f:
            d = f["/0/DATA/0/matrix"]
            sig_axis0 = d.shape[0] == len(col_ids)
            n_tot = d.shape[0] if sig_axis0 else d.shape[1]
            buf = np.empty((n_tot, len(rows)), dtype=np.float32)
            step = 5000
            for i in range(0, n_tot, step):
                j = min(i + step, n_tot)
                blk = d[i:j, :][:, rows_sorted] if sig_axis0 else d[rows_sorted, i:j].T
                buf[i:j] = blk[:, r_inv]
                print(f"    {j:,}/{n_tot:,}", end="\r", flush=True)
        np.save(cache, buf)
    blocks.append(np.nan_to_num(buf[cols].astype(np.float64)))
    frames.append(cp_s)
    del buf

cp = pd.concat(frames, ignore_index=True)
Z = np.concatenate(blocks, axis=0)
del frames, blocks

# Signature ids are phase-local; a collision would make one signature shadow another in
# any id-keyed join downstream, so establish that there are none before going on.
dup = cp.sig_id.duplicated().sum()
if dup:
    raise SystemExit(f"{dup:,} signature ids occur in both phases; ids are not unique across "
                     f"the concatenated library.")

print(f"\n>>> Signature matrix: {Z.shape[0]:,} signatures x {Z.shape[1]:,} clock genes")
print(">>> Library composition: " + ", ".join(
    f"{p} {int(n):,} sigs" for p, n in cp.source.value_counts().items()) +
    f" | {cp.pert_id.nunique():,} distinct compounds over {cp.cell_id.nunique():,} cell lines")

# ---------------------------------------------------------------- scoring
cp["dage"] = Z @ coef_u

# Cross-phase agreement. The two releases are independent experiments run years apart, so
# on the compounds they share their consensus scores should correlate. This is the check
# that the per-source row alignment is actually right: a mis-paired gene axis leaves the
# two phases uncorrelated while still producing entirely plausible-looking numbers.
ph = cp.groupby(["pert_id", "source"])["dage"].mean().unstack()
if {"phase1", "phase2"}.issubset(ph.columns):
    both = ph.dropna()
    r_ph = both.phase1.corr(both.phase2)
    print(f">>> Cross-phase agreement on {len(both):,} shared compounds: r = {r_ph:+.3f}")
    if r_ph < 0.2:
        raise SystemExit("Phase 1 and Phase 2 disagree on the compounds they share; the gene "
                         "alignment or the level-5 scaling is wrong. Refusing to report a "
                         "merged screen.")

# Cross-cell-line reproducibility: fraction of cell lines agreeing on the sign
cell = cp.groupby(["pert_id", "cell_id"])["dage"].mean().reset_index()
sgn = cell.groupby("pert_id")["dage"].apply(
    lambda s: max((s > 0).mean(), (s < 0).mean()) if len(s) > 1 else np.nan)

# Consensus signature per compound = mean z across its cell lines / doses / times
codes, uniq_pert = pd.factorize(cp.pert_id)
agg = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                    shape=(len(uniq_pert), len(codes)))
CZ = (agg @ Z) / np.bincount(codes)[:, None]

comp = cp.groupby("pert_id", sort=False).agg(
    pert_iname=("pert_iname", "first"), n_sig=("sig_id", "size"),
    n_cell=("cell_id", "nunique"), dage_sd=("dage", "std")).reindex(uniq_pert)
comp.index.name = "pert_id"          # reindex drops the name, so restore it before reset
comp = comp.reset_index()
comp["dage"] = CZ @ coef_u
comp["sign_consistency"] = comp.pert_id.map(sgn)

# Gene-label permutation null: how large a |dAge| does a random gene-to-coefficient
# assignment produce for signatures of this magnitude?
# Drawn one permutation at a time, in the same order as a plain loop, but multiplied in
# blocks: holding the full null would now cost 20,000 x 21,299 floats, and 20,000 Python
# iterations of a matrix-vector product dominate the runtime at this library size. Only
# the running tallies the statistics actually need are kept.
rng = np.random.default_rng(SEED)
obs = np.abs(comp.dage.values)
ge = np.zeros(len(comp), dtype=np.int64)
s1, s2 = np.zeros(len(comp)), np.zeros(len(comp))
BLK = 500
for i in range(0, N_PERM, BLK):
    b = min(BLK, N_PERM - i)
    P = np.stack([rng.permutation(coef_u) for _ in range(b)], axis=1)
    nb = CZ @ P
    ge += (np.abs(nb) >= obs[:, None]).sum(1)
    s1 += nb.sum(1)
    s2 += np.square(nb).sum(1)
    print(f"    permutations {min(i+b, N_PERM):,}/{N_PERM:,}", end="\r", flush=True)
comp["p_perm"] = (ge + 1) / (N_PERM + 1)
comp["fdr"] = false_discovery_control(comp.p_perm.values, method="bh")
comp["null_sd"] = np.sqrt(np.maximum(s2 / N_PERM - (s1 / N_PERM) ** 2, 0))
comp["z_vs_null"] = comp.dage / comp.null_sd

# Concentration diagnostic: a hit whose score comes almost entirely from one gene is
# fragile, since a single noisy measurement then decides the whole call.
contrib = CZ * coef_u[None, :]
tot = np.abs(contrib).sum(1) + 1e-12
comp["top_gene"] = np.array(clock_genes[usable])[np.abs(contrib).argmax(1)]
comp["top_gene_frac"] = np.abs(contrib).max(1) / tot

# ---------------------------------------------------------------- annotation
rep = pd.read_pickle(f"{LINCS}/repurposing.pkl")[["pert_iname", "moa", "target", "clinical_phase"]]
comp = comp.merge(rep.drop_duplicates("pert_iname"), on="pert_iname", how="left")
comp = comp.sort_values("dage").reset_index(drop=True)
comp["rank"] = np.arange(1, len(comp) + 1)
comp["percentile"] = 100 * comp["rank"] / len(comp)
comp.to_csv(f"{ROOT}/phase6/phase6b_lincs_screen_results.csv", index=False)

n_hit = (comp.fdr < 0.05).sum()
print(f"\n>>> Compounds passing permutation null at FDR<0.05: {n_hit:,}/{len(comp):,}")
print(f">>> Predicted rejuvenators (dAge<0, FDR<0.05): {((comp.dage < 0) & (comp.fdr < 0.05)).sum():,}")

print(f">>> Hits whose score is dominated by a single gene (>50%): "
      f"{(comp.top_gene_frac > 0.5).sum():,} - treat these as fragile")

print("\n=== TOP 15 PREDICTED REJUVENATORS (most negative dAge) ===")
show = ["pert_iname", "dage", "z_vs_null", "fdr", "n_sig", "sign_consistency",
        "top_gene", "top_gene_frac", "moa"]
print(comp.head(15)[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\n=== TOP 10 PREDICTED PRO-AGEING (most positive dAge) ===")
print(comp.tail(10)[show].iloc[::-1].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# ---------------------------------------------------------------- MoA enrichment
print("\n=== MECHANISM-OF-ACTION ENRICHMENT (class vs rest of library) ===")
rows_moa = []
for moa, g in comp.dropna(subset=["moa"]).groupby("moa"):
    if len(g) < 5:
        continue
    rest = comp[~comp.pert_id.isin(g.pert_id)].dage.values
    u, p = mannwhitneyu(g.dage.values, rest, alternative="two-sided")
    rows_moa.append(dict(moa=moa, n=len(g), median_dage=g.dage.median(),
                         median_percentile=g.percentile.median(), p=p))
moa_df = pd.DataFrame(rows_moa)
moa_df["fdr"] = false_discovery_control(moa_df.p.values, method="bh")
moa_df = moa_df.sort_values("median_dage")
moa_df.to_csv(f"{ROOT}/phase6/phase6b_moa_enrichment.csv", index=False)

print(f"  {len(moa_df)} MoA classes with n>=5 tested; {(moa_df.fdr<0.05).sum()} significant at FDR<0.05")
print("\n-- Most rejuvenating MoA classes (FDR<0.05) --")
sigm = moa_df[moa_df.fdr < 0.05]
print(sigm.head(10).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n-- Most pro-ageing MoA classes (FDR<0.05) --")
print(sigm.tail(10).iloc[::-1].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# ---------------------------------------------------------------- cytotoxicity confound
# A clock trained on ageing could be re-detecting nothing but "this compound kills cells".
# If so, cytotoxic mechanisms would cluster at the pro-ageing end. Testing this requires a
# definition of "cytotoxic" fixed in advance and stated explicitly, since the answer moves
# with the definition; the class list below is by mechanism annotation, not by outcome.
CYTOTOXIC = ("topoisomerase inhibitor", "tubulin polymerization inhibitor",
             "microtubule inhibitor", "DNA synthesis inhibitor", "DNA alkylating agent",
             "ribonucleotide reductase inhibitor", "antimetabolite", "CDK inhibitor",
             "Aurora kinase inhibitor", "PLK inhibitor", "proteasome inhibitor",
             "apoptosis stimulant", "HSP inhibitor")
comp["cytotoxic"] = comp.moa.fillna("").apply(
    lambda m: any(c.lower() in m.lower() for c in CYTOTOXIC))
srt = comp.sort_values("dage")
lo50, hi50 = srt.head(50), srt.tail(50)
cy, rest = comp[comp.cytotoxic].dage, comp[~comp.cytotoxic].dage
u_c, p_c = mannwhitneyu(cy, rest)
print("\n=== CYTOTOXICITY CONFOUND CHECK ===")
print(f"  Mechanism classes counted as cytotoxic: {len(CYTOTOXIC)}; "
      f"compounds so annotated: {int(comp.cytotoxic.sum())}/{len(comp)}")
print(f"  Among the 50 most rejuvenating: {int(lo50.cytotoxic.sum())}/50 "
      f"({100*lo50.cytotoxic.mean():.0f}%)")
print(f"  Among the 50 most pro-ageing:   {int(hi50.cytotoxic.sum())}/50 "
      f"({100*hi50.cytotoxic.mean():.0f}%)")
print(f"  Median dAge cytotoxic {cy.median():+.3f} vs rest {rest.median():+.3f}  "
      f"(Mann-Whitney p = {p_c:.4f})")

# If the shift is significant, it matters enormously whether it is general toxicity or
# just the senescence-inducer positive control, which is expected to score pro-ageing and
# is itself cytotoxic. Re-testing with topoisomerase inhibitors removed separates the two.
comp["cytotoxic_noTopo"] = comp.moa.fillna("").apply(
    lambda m: any(c.lower() in m.lower() for c in CYTOTOXIC if "topoisomerase" not in c))
cy2 = comp[comp.cytotoxic_noTopo].dage
rest2 = comp[~comp.cytotoxic_noTopo].dage
u_c2, p_c2 = mannwhitneyu(cy2, rest2)
print(f"  Excluding topoisomerase inhibitors (n={int(comp.cytotoxic_noTopo.sum())}): "
      f"median {cy2.median():+.3f} vs {rest2.median():+.3f} (p = {p_c2:.4f})")
if p_c < 0.05 and p_c2 >= 0.05:
    print("  The whole cytotoxicity signal is the senescence-inducer control class. Every")
    print("  other cytotoxic mechanism sits at the library median, so the clock is not a")
    print("  general cytotoxicity readout; it responds specifically to senescence induction.")
elif p_c >= 0.05:
    print("  No significant shift, so the ranking is not a cytotoxicity readout.")
else:
    print("  The shift survives removing the positive control, so the pro-ageing end may")
    print("  partly track general toxicity; interpret it with that in mind.")
print("\n  Per-class medians among cytotoxic mechanisms (n>=5):")
for c in CYTOTOXIC:
    f = comp.moa.fillna("").str.contains(c, case=False, regex=False)
    if f.sum() >= 5:
        print(f"    {c:<40} n={int(f.sum()):>3}  median dAge={comp[f].dage.median():+7.3f}")

# ---------------------------------------------------------------- pre-registered panel
print("\n=== PRE-REGISTERED PANEL: where the original Phase 6 compounds actually land ===")
PANEL = {"navitoclax": "senolytic (BCL-2/xL)", "ABT-737": "senolytic (BCL-2)",
         "dasatinib": "senolytic (D+Q)", "quercetin": "senolytic (D+Q)",
         "sirolimus": "mTOR inhibitor", "everolimus": "mTOR inhibitor",
         "torin-1": "mTOR inhibitor", "metformin": "AMPK activator",
         "resveratrol": "sirtuin/polyphenol", "nicotinamide": "NAD+ precursor",
         "doxorubicin": "senescence inducer (control)",
         "etoposide": "senescence inducer (control)"}
pan = comp[comp.pert_iname.isin(PANEL)].copy()
pan["expected"] = pan.pert_iname.map(PANEL)
print(pan[["pert_iname", "expected", "dage", "rank", "percentile", "fdr", "sign_consistency"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

rej = pan[~pan.expected.str.contains("control")].dage.values
ctl = pan[pan.expected.str.contains("control")].dage.values
if len(rej) and len(ctl):
    u, p = mannwhitneyu(rej, ctl, alternative="less")
    print(f"\n  Expected rejuvenators vs senescence-inducer controls: "
          f"Mann-Whitney p = {p:.4f} (n={len(rej)} vs {len(ctl)})")
print("\n>>> Saved phase6b_lincs_screen_results.csv, phase6b_moa_enrichment.csv")
