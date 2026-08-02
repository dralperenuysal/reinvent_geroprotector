#!/usr/bin/env python
"""
Plan B, step 1: LINCS L1000 connectivity screen against the SenMayo signature.

Adapted from aging_clock's phase6b_lincs_screen.py. Instead of a trained clock's
per-gene coefficients, the "readout" here is simply the 119/125 SenMayo genes
measured or best-inferred by L1000, uniformly weighted. A compound's reversal
score is how far its SenMayo genes sit below (down-regulated relative to) a
background of same-size random gene sets drawn from the same measured pool, in
the compound's consensus signature -- i.e. exactly gps_oracle.py's permutation-Z
methodology, but computed from real measured LINCS data instead of GPS's
per-gene neural-network predictions.

This script exists because three independent GPS-based scoring methodologies
all failed the Section 3.0 positive-control robustness check, and a targeted
diagnostic (oracle/real_lincs_check.py) showed the failure was specific to
GPS's predictions: the real measured LINCS data does not show the same
systematic "SenMayo always up" bias (library mean SenMayo z ~ 0, not strongly
positive). This screen is the full, honest version of that diagnostic: every
compound scored, with a proper permutation null, cross-phase agreement check,
and cytotoxicity confound check, instead of just a hand-picked comparison.
"""
import os, warnings
import numpy as np, pandas as pd, h5py
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, false_discovery_control

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINCS = f"{ROOT}/data/raw/lincs"
SENMAYO_PATH = f"{ROOT}/data/raw/senmayo/senmayo_human_genes.txt"
OUT = f"{ROOT}/data/processed"
os.makedirs(OUT, exist_ok=True)

N_PERM = 2000
SEED = 0
POSITIVE_CONTROLS = ["dasatinib", "navitoclax", "panobinostat", "quercetin", "sirolimus", "vorinostat"]

SOURCES = [dict(name="GSE70138", phase="phase2", gctx=f"{LINCS}/L5_GSE70138.gctx",
                sig=f"{LINCS}/sig_info70.txt", gi=f"{LINCS}/gene_info70.txt"),
           dict(name="GSE92742", phase="phase1", gctx=f"{LINCS}/L5_GSE92742.gctx",
                sig=f"{LINCS}/sig_info92.txt", gi=f"{LINCS}/gene_info92.txt")]

print("=== PLAN B STEP 1: LINCS L1000 SCREEN vs SenMayo SIGNATURE (real measured data) ===")

senmayo_genes = [l.strip() for l in open(SENMAYO_PATH)]
print(f">>> SenMayo genes (target signature): {len(senmayo_genes)}")


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
    raise SystemExit(f"Missing level-5 matrices for {', '.join(absent)}.")

# Background gene pool: BING genes ("best inferred", 1.0 confidence) from the merged
# gene_info.pkl, which both phases share (same 12,328-gene, same Entrez-id space --
# see phase6b's own comment on this). This mirrors gps_oracle.py's background choice
# so the two oracles are directly comparable.
gi = pd.read_pickle(f"{LINCS}/gene_info.pkl")
bg_genes = sorted(gi[gi.pr_is_bing == 1].pr_gene_symbol)
senmayo_in_bg = sorted(set(senmayo_genes) & set(bg_genes))
print(f">>> Background (BING, both phases): {len(bg_genes):,} genes | "
      f"SenMayo covered: {len(senmayo_in_bg)}/{len(senmayo_genes)}")

id2sym92 = dict(zip(gi.pr_gene_id.astype(str), gi.pr_gene_symbol.astype(str)))

# Two passes to avoid ever holding two full-size copies of the (n_signatures x 10,174)
# matrix at once (a plain per-source list + np.concatenate at the end briefly needs both
# the per-source blocks AND the concatenated array alive simultaneously -- ~25GB against
# ~27GB available, which OOM-killed the first attempt with no traceback). Pass 1 only
# reads the small sig_info tables to work out row counts; pass 2 fills one preallocated
# array directly, one source's cache load at a time.
rows = usable = None
meta = []
for src in SOURCES:
    sig = pd.read_csv(src["sig"], sep="\t", low_memory=False)
    with h5py.File(src["gctx"], "r") as f:
        row_ids = as_str(f["/0/META/ROW/id"][:])
        col_ids = as_str(f["/0/META/COL/id"][:])
        dshape = f["/0/DATA/0/matrix"].shape
    print(f"\n>>> {src['name']} ({src['phase']}): matrix {dshape} | "
          f"{len(row_ids):,} genes x {len(col_ids):,} signatures")

    row_sym = np.array([id2sym92.get(r, "") for r in row_ids])
    sym2row = {s: i for i, s in enumerate(row_sym) if s}
    usable_s = np.array([g in sym2row for g in bg_genes])
    if usable_s.sum() < 0.9 * len(bg_genes):
        raise SystemExit(f"Only {usable_s.sum()}/{len(bg_genes)} background genes matched "
                         f"{src['name']} rows - gene id mapping is broken, refusing to score.")
    if usable is None:
        usable = usable_s
    elif not np.array_equal(usable_s, usable):
        raise SystemExit(f"{src['name']} measures a different subset of background genes than "
                         f"{SOURCES[0]['name']}; the concatenated columns would not align.")
    rows = np.array([sym2row[g] for g in np.array(bg_genes)[usable]])

    sig_idx = {s: i for i, s in enumerate(col_ids)}
    cp_s = sig[(sig.pert_type == "trt_cp") & sig.sig_id.isin(sig_idx)].copy()
    cp_s["source"] = src["phase"]
    cols = np.array([sig_idx[s] for s in cp_s.sig_id])
    print(f">>> Compound signatures: {len(cp_s):,} across {cp_s.pert_id.nunique():,} compounds")
    meta.append(dict(src=src, cp_s=cp_s, cols=cols, col_ids=col_ids, rows=rows))

total_rows = sum(len(m["cols"]) for m in meta)
Z = np.empty((total_rows, len(rows)), dtype=np.float32)
frames = []
offset = 0
for m in meta:
    src, cp_s, cols, col_ids, rows = m["src"], m["cp_s"], m["cols"], m["col_ids"], m["rows"]
    n = len(cols)
    dest = Z[offset:offset + n]

    r_ord = np.argsort(rows)
    rows_sorted, r_inv = rows[r_ord], np.argsort(r_ord)
    cache = f"{LINCS}/slab_{src['name']}_senmayo_bg_{len(rows)}g_trtcp.npy"
    if os.path.exists(cache):
        print(f">>> Reusing cached expression slab ({os.path.basename(cache)})")
        cached = np.load(cache, mmap_mode="r")
        dest[:] = cached
        del cached
    else:
        print(">>> Reading expression slab from GCTX (compound signatures only)...", flush=True)
        col_out_pos = np.full(len(col_ids), -1, dtype=np.int64)
        col_out_pos[cols] = np.arange(n)
        with h5py.File(src["gctx"], "r") as f:
            d = f["/0/DATA/0/matrix"]
            sig_axis0 = d.shape[0] == len(col_ids)
            n_tot = d.shape[0] if sig_axis0 else d.shape[1]
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
                dest[out_pos[keep]] = blk[keep]
                print(f"    {j:,}/{n_tot:,}", end="\r", flush=True)
        np.save(cache, dest)
    np.nan_to_num(dest, copy=False)
    frames.append(cp_s)
    offset += n

cp = pd.concat(frames, ignore_index=True)
del frames, meta

dup = cp.sig_id.duplicated().sum()
if dup:
    raise SystemExit(f"{dup:,} signature ids occur in both phases; ids are not unique across "
                     f"the concatenated library.")

bg_used = np.array(bg_genes)[usable]
senmayo_idx = np.array([i for i, g in enumerate(bg_used) if g in set(senmayo_in_bg)])
print(f"\n>>> Signature matrix: {Z.shape[0]:,} signatures x {Z.shape[1]:,} background genes "
      f"({len(senmayo_idx)} SenMayo)")
print(">>> Library composition: " + ", ".join(
    f"{p} {int(n):,} sigs" for p, n in cp.source.value_counts().items()) +
    f" | {cp.pert_id.nunique():,} distinct compounds over {cp.cell_id.nunique():,} cell lines")

# ---------------------------------------------------------------- scoring
# Raw signed score per signature: mean SenMayo z (positive = SenMayo up = bad).
cp["senmayo_mean_z"] = Z[:, senmayo_idx].mean(axis=1)

ph = cp.groupby(["pert_id", "source"])["senmayo_mean_z"].mean().unstack()
if {"phase1", "phase2"}.issubset(ph.columns):
    both = ph.dropna()
    r_ph = both.phase1.corr(both.phase2)
    print(f">>> Cross-phase agreement on {len(both):,} shared compounds: r = {r_ph:+.3f}")
    if r_ph < 0.15:
        raise SystemExit("Phase 1 and Phase 2 disagree on the compounds they share; the gene "
                         "alignment or the level-5 scaling is wrong. Refusing to report a "
                         "merged screen.")

cell = cp.groupby(["pert_id", "cell_id"])["senmayo_mean_z"].mean().reset_index()
sgn = cell.groupby("pert_id")["senmayo_mean_z"].apply(
    lambda s: max((s > 0).mean(), (s < 0).mean()) if len(s) > 1 else np.nan)

# Consensus signature per compound (mean across all its signatures/cell lines/doses/times).
# agg's nonzero values must be float32 (matching Z), not scipy's float64 default: a mixed-
# dtype sparse @ dense matmul makes scipy upcast the ENTIRE dense operand to float64 first,
# i.e. a transient full-size float64 copy of Z on top of Z itself (~24GB + ~12GB) -- almost
# certainly what OOM-killed the previous attempt despite Z alone fitting comfortably.
codes, uniq_pert = pd.factorize(cp.pert_id)
agg = sp.csr_matrix((np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
                    shape=(len(uniq_pert), len(codes)), dtype=np.float32)
CZ = (agg @ Z) / np.bincount(codes)[:, None].astype(np.float32)   # (n_compounds, n_bg_genes)

comp = cp.groupby("pert_id", sort=False).agg(
    pert_iname=("pert_iname", "first"), n_sig=("sig_id", "size"),
    n_cell=("cell_id", "nunique")).reindex(uniq_pert)
comp.index.name = "pert_id"
comp = comp.reset_index()
comp["senmayo_mean_z"] = CZ[:, senmayo_idx].mean(axis=1)
comp["sign_consistency"] = comp.pert_id.map(sgn)

# Permutation null: same-size random gene sets drawn from the background pool, for
# each compound's consensus profile. Reversal score is the sign-flipped Z of the
# observed SenMayo mean against this null (positive = SenMayo pushed down relative
# to a random gene set of the same size = desired reversal direction).
rng = np.random.default_rng(SEED)
n_bg = len(bg_used)
n_sm = len(senmayo_idx)
n_comp = CZ.shape[0]
perm_idx = np.stack([rng.choice(n_bg, size=n_sm, replace=False) for _ in range(N_PERM)])
# CZ[:, perm_idx].mean(axis=2) would materialize a dense (n_compounds, N_PERM, n_sm) tensor
# -- at this library's scale (~22k compounds) that's tens of GB and OOM-kills the process.
# The mean-over-permutation-genes is a linear map, so express it as a sparse matmul instead:
# null_means[i, p] = mean_g CZ[i, g] for g in perm_idx[p], i.e. CZ @ M.T / n_sm where M is a
# (N_PERM x n_bg) 0/1 indicator with n_sm ones per row.
perm_rows = np.repeat(np.arange(N_PERM), n_sm)
M = sp.csr_matrix((np.ones(perm_rows.size, dtype=np.float32), (perm_rows, perm_idx.ravel())),
                  shape=(N_PERM, n_bg), dtype=np.float32)
null_means_T = (M @ CZ.T) / n_sm                   # (N_PERM, n_compounds), dense but small
null_mean = null_means_T.mean(axis=0)
null_std = null_means_T.std(axis=0)
null_std = np.where(null_std < 1e-8, 1e-8, null_std)
comp["z_vs_null"] = (comp.senmayo_mean_z.values - null_mean) / null_std
comp["reversal_score"] = -comp["z_vs_null"]        # sign-flip: down-regulation = good = positive

# Empirical two-sided p-value from the same null draws (cheap upper bound, N_PERM+1 floor)
obs = comp.senmayo_mean_z.values
ge = (np.abs(null_means_T - null_mean[None, :]) >= np.abs(obs - null_mean)[None, :]).sum(axis=0)
comp["p_perm"] = (ge + 1) / (N_PERM + 1)
comp["fdr"] = false_discovery_control(comp.p_perm.values, method="bh")

# ---------------------------------------------------------------- annotation
rep = pd.read_pickle(f"{LINCS}/repurposing.pkl")[["pert_iname", "moa", "target", "clinical_phase"]]
comp = comp.merge(rep.drop_duplicates("pert_iname"), on="pert_iname", how="left")
comp = comp.sort_values("reversal_score", ascending=False).reset_index(drop=True)
comp["rank"] = np.arange(1, len(comp) + 1)
comp["percentile"] = 100 * (1 - comp["rank"] / len(comp))
comp.to_csv(f"{OUT}/lincs_senmayo_screen_results.csv", index=False)

n_hit = (comp.fdr < 0.05).sum()
print(f"\n>>> Compounds passing permutation null at FDR<0.05: {n_hit:,}/{len(comp):,}")
print(f">>> Predicted reversers (reversal_score>0, FDR<0.05): "
      f"{((comp.reversal_score > 0) & (comp.fdr < 0.05)).sum():,}")

print("\n=== TOP 15 PREDICTED SenMayo REVERSERS ===")
show = ["pert_iname", "reversal_score", "z_vs_null", "fdr", "n_sig", "sign_consistency", "moa"]
print(comp.head(15)[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\n=== BOTTOM 10 (predicted SenMayo INDUCERS) ===")
print(comp.tail(10)[show].iloc[::-1].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# ---------------------------------------------------------------- positive-control panel
print("\n=== SECTION 3.0 ROBUSTNESS CHECK: known senolytics/senomorphics (real LINCS data) ===")
pc = comp[comp.pert_iname.str.lower().isin(POSITIVE_CONTROLS)]
lib_mean = comp.reversal_score.mean()
above = (pc.reversal_score > lib_mean).sum()
print(pc[["pert_iname", "reversal_score", "percentile", "fdr"] ].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(f"\nLibrary mean reversal_score = {lib_mean:+.4f}")
print(f"{above}/{len(pc)} positive controls score above the library mean.")

# ---------------------------------------------------------------- cytotoxicity confound
CYTOTOXIC = ("topoisomerase inhibitor", "tubulin polymerization inhibitor",
             "microtubule inhibitor", "DNA synthesis inhibitor", "DNA alkylating agent",
             "ribonucleotide reductase inhibitor", "antimetabolite", "CDK inhibitor",
             "Aurora kinase inhibitor", "PLK inhibitor", "proteasome inhibitor",
             "apoptosis stimulant", "HSP inhibitor")
comp["cytotoxic"] = comp.moa.fillna("").apply(lambda m: any(c.lower() in m.lower() for c in CYTOTOXIC))
cy, rest = comp[comp.cytotoxic].reversal_score, comp[~comp.cytotoxic].reversal_score
if comp.cytotoxic.sum() >= 5:
    u_c, p_c = mannwhitneyu(cy, rest)
    print("\n=== CYTOTOXICITY CONFOUND CHECK ===")
    print(f"  Cytotoxic-annotated compounds: {int(comp.cytotoxic.sum())}/{len(comp)}")
    print(f"  Median reversal_score cytotoxic {cy.median():+.3f} vs rest {rest.median():+.3f} "
          f"(Mann-Whitney p = {p_c:.4f})")

print("\n>>> Saved data/processed/lincs_senmayo_screen_results.csv")
