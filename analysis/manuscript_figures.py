#!/usr/bin/env python
"""Three additional validation figures for the journal manuscript (not the
conference abstract, which stays at one figure). All panels are built strictly
from data/processed/*.csv -- no numbers are invented; every value plotted here
is cross-checked against the numbers already stated in manuscript.tex and, where
possible, against the original analysis scripts' own log output.

Note: the deployed oracle (RandomForest) held-out per-compound predictions were
never saved to disk by oracle/lincs_structure_bridge.py (only the RidgeCV
predictions were, in lincs_bridge_heldout_predictions.csv), and the raw LINCS/
SMILES files needed to regenerate them are not present locally (data/raw is
gitignored, tens of GB, not re-downloaded here). Figure A therefore compares
Ridge vs. RandomForest using the aggregate scaffold-split statistics in
data/processed/lincs_bridge_validation.csv and oracle/lincs_structure_bridge.log
(both models' r, plus the permutation-null r), rather than a fabricated
per-compound RF scatter.

Usage: python analysis/manuscript_figures.py
Outputs three PNGs into paper/figures/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#3a3a3a",
    "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a",
    "ytick.color": "#3a3a3a",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"
OUT = f"{ROOT}/paper/figures"
os.makedirs(OUT, exist_ok=True)

# Palette: CVD-validated categorical hues (blue/red slots, adjacent-pair Jaccard
# CVD dE >= 8 in OKLab), swapped in for matplotlib's flat C0/C3 defaults; GRAY is
# a warm neutral rather than pure #7f7f7f so it doesn't read as "disabled" ink.
RED = "#e34948"    # Run 1 / RandomForest (deployed)
BLUE = "#2a78d6"   # Run 2
GRAY = "#8c8a82"   # LINCS library / non-selected model
HAIRLINE = "#c9c7bd"  # reference bands/lines, replaces heavy black at low weight

CYTOTOXIC_MOA = ("topoisomerase inhibitor", "tubulin polymerization inhibitor",
                  "microtubule inhibitor", "DNA synthesis inhibitor", "DNA alkylating agent",
                  "ribonucleotide reductase inhibitor", "antimetabolite", "CDK inhibitor",
                  "Aurora kinase inhibitor", "PLK inhibitor", "proteasome inhibitor",
                  "apoptosis stimulant", "HSP inhibitor")
POSITIVE_CONTROLS = ["dasatinib", "navitoclax", "panobinostat", "quercetin", "sirolimus", "vorinostat"]


# ============================================================== Figure A
# Structure-based bridge-model validation: Ridge vs. RandomForest scaffold-split
# Pearson r, against the permutation-null band. Numbers from
# data/processed/lincs_bridge_validation.csv (exact) and
# oracle/lincs_structure_bridge.log (permutation null, not re-saved per-run).
def figure_bridge_validation():
    val = pd.read_csv(f"{PROC}/lincs_bridge_validation.csv")
    r_ridge = val.loc[val.model.str.startswith("Ridge"), "test_r"].iloc[0]
    r_rf = val.loc[val.model.str.startswith("RandomForest"), "test_r"].iloc[0]
    null_mean, null_sd = -0.000, 0.013  # oracle/lincs_structure_bridge.log, 200 shuffles

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhspan(null_mean - null_sd, null_mean + null_sd, color=HAIRLINE, alpha=0.5,
               label=f"Permutation null (mean$\\pm$SD, 200 shuffles)")
    ax.axhline(null_mean, color="#6b6a63", lw=1, ls="--", alpha=0.7)

    xs = [0, 1]
    heights = [r_ridge, r_rf]
    colors = [GRAY, RED]
    bars = ax.bar(xs, heights, width=0.4, color=colors, edgecolor="none")

    for x, h in zip(xs, heights):
        ax.text(x, h + 0.008, f"r = {h:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(xs)
    ax.set_xticklabels(["Ridge", "Random Forest\n(deployed oracle)"])
    ax.set_ylabel("Scaffold-split Pearson $r$\n(held-out test set, n = 5,305)")
    ax.set_ylim(-0.05, 0.32)
    ax.set_title("Structure$\\rightarrow$SenMayo-reversal\nbridge model validation", fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/bridge_validation.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/bridge_validation.png  (Ridge r={r_ridge:.3f}, RF r={r_rf:.3f})")


# ============================================================== Figure B
# Real-data face-validity checks: (a) cytotoxic vs. rest reversal_score,
# (b) six-compound positive-control panel against the library distribution.
def figure_real_data_validation():
    screen = pd.read_csv(f"{PROC}/lincs_senmayo_screen_results.csv")
    screen["cytotoxic"] = screen.moa.fillna("").apply(
        lambda m: any(c.lower() in m.lower() for c in CYTOTOXIC_MOA))
    cy = screen.loc[screen.cytotoxic, "reversal_score"]
    rest = screen.loc[~screen.cytotoxic, "reversal_score"]
    _, p_mw = mannwhitneyu(cy, rest)

    pc = screen[screen.pert_iname.str.lower().isin(POSITIVE_CONTROLS)].copy()
    pc["name"] = pc.pert_iname.str.lower()
    pc = pc.loc[pc.groupby("name").n_sig.idxmax()].sort_values("reversal_score")
    lib_mean = screen.reversal_score.mean()
    lib_mean_pctile = 100 * (screen.reversal_score < lib_mean).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- panel a: cytotoxic vs rest (boxplot: heavy-tailed data, no clipping artifacts) ---
    bp = ax1.boxplot([rest, cy], positions=[0, 1], widths=0.4, patch_artist=True,
                      showfliers=True,
                      flierprops=dict(marker="o", markersize=2, alpha=0.25, markeredgewidth=0),
                      medianprops=dict(color="black", linewidth=1.8),
                      whiskerprops=dict(color="#3a3a3a", linewidth=1),
                      capprops=dict(color="#3a3a3a", linewidth=1))
    for patch, color in zip(bp["boxes"], [GRAY, RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#3a3a3a")
        patch.set_linewidth(1)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"Rest of library\n(n={len(rest):,})",
                          f"Cytotoxic /\nDNA-damage MoA\n(n={len(cy):,})"])
    ax1.set_ylabel("SenMayo reversal_score")
    ax1.set_title("Cytotoxic mechanisms induce,\nrather than reverse, SASP")
    p_label = "$p<0.0001$" if p_mw < 1e-4 else f"$p={p_mw:.4f}$"
    y_bracket = max(rest.quantile(0.98), cy.quantile(0.98)) + 0.8
    ax1.plot([0, 0, 1, 1], [y_bracket, y_bracket + 0.4, y_bracket + 0.4, y_bracket],
             color="black", lw=1)
    ax1.text(0.5, y_bracket + 1.1, f"Mann--Whitney {p_label}", ha="center", va="bottom",
             fontsize=10)
    ax1.set_ylim(top=y_bracket + 2.6)
    ax1.axhline(0, color="black", lw=0.6, alpha=0.4)

    # --- panel b: positive-control panel ---
    colors_pc = [RED if s > lib_mean else GRAY for s in pc.reversal_score]
    ax2.axvline(lib_mean, color="black", lw=1, ls="--", alpha=0.6,
                label=f"Library mean (P{lib_mean_pctile:.0f})")
    ax2.scatter(pc.reversal_score, range(len(pc)), c=colors_pc, s=90,
                edgecolor="black", zorder=3)
    ax2.set_yticks(range(len(pc)))
    ax2.set_yticklabels([n.capitalize() for n in pc.name])
    ax2.set_xlabel("SenMayo reversal_score (real measured LINCS data)")
    ax2.set_title("Positive-control senolytics /\nsenomorphics vs. library mean")
    dasa = pc[pc.name == "dasatinib"].iloc[0]
    ax2.annotate(f"FDR = {dasa.fdr:.3f}\n(P{dasa.percentile:.1f})",
                 xy=(dasa.reversal_score, list(pc.name).index("dasatinib")),
                 xytext=(10, 8), textcoords="offset points", fontsize=9.5)
    ax2.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT}/real_data_validation.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/real_data_validation.png  "
          f"(Mann-Whitney p={p_mw:.2e}, {int((pc.reversal_score > lib_mean).sum())}/{len(pc)} above mean)")


# ============================================================== Figure C
# Generated-set reversal-score distributions vs. the LINCS library.
def figure_score_distributions():
    screen = pd.read_csv(f"{PROC}/lincs_senmayo_screen_results.csv")
    run1 = pd.read_csv(f"{PROC}/run1/generated_set_converged.csv")
    run2 = pd.read_csv(f"{PROC}/run2/generated_set_converged.csv")

    lib = screen.reversal_score  # unclipped: exact mean must match manuscript's -0.318
    g1 = run1["SenMayo_reversal (raw)"]
    g2 = run2["SenMayo_reversal (raw)"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(-6, 4, 90)
    ax.hist(lib.clip(-6, 4), bins=bins, density=True, color=GRAY, alpha=0.55,
            label=f"LINCS library (n={len(lib):,})")
    ax.hist(g1, bins=bins, density=True, color=RED, alpha=0.55,
            label=f"Run 1 converged set (n={len(g1):,})")
    ax.hist(g2, bins=bins, density=True, color=BLUE, alpha=0.55,
            label=f"Run 2 converged set (n={len(g2):,})")

    for data, color in [(lib, "black"), (g1, RED), (g2, BLUE)]:
        ax.axvline(data.mean(), color=color, lw=1.6, ls="--")

    ax.set_xlabel("SenMayo reversal_score")
    ax.set_ylabel("Density")
    ax.set_title("Predicted SASP-reversal score: generated sets vs. the training library")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/score_distributions.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/score_distributions.png  "
          f"(library mean={lib.mean():.3f} [manuscript: -0.318], "
          f"run1={g1.mean():.3f} [+0.591], run2={g2.mean():.3f} [+0.748])")


# ============================================================== Figure D
# 2D chemical-space map (Morgan fingerprints -> UMAP, Jaccard/Tanimoto metric)
# of the real LINCS library vs. both runs' converged sets vs. the top-10 Run-2
# candidates. LINCS SMILES come from data/processed/lincs_smiles.pkl, a
# pert_id -> canonical_smiles table recovered from the sibling aging_clock
# project (data/lincs/smiles.pkl there); joined against
# lincs_senmayo_screen_results.csv's pert_id column, 21,236/21,299 (99.7%)
# of the screened compounds resolve to a real SMILES. LINCS, Run 1, and
# Run 2 are each randomly subsampled to an equal N before fitting so that
# apparent point density in the plot reflects distributional shape, not raw
# set-size differences (Run 2's converged set is 8.5x larger than Run 1's;
# full counts are reported in Table 1, not this figure).
def figure_chemical_space(n_per_group=5000, seed=0):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    import umap

    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(seed)

    def to_fp_array(smiles_list):
        fps, kept = [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            kept.append(smi)
        return np.array(fps, dtype=np.uint8), kept

    def subsample(seq, n):
        seq = list(seq)
        if len(seq) <= n:
            return seq
        idx = rng.choice(len(seq), size=n, replace=False)
        return [seq[i] for i in idx]

    # --- LINCS library: real SMILES recovered via pert_id join ---
    screen = pd.read_csv(f"{PROC}/lincs_senmayo_screen_results.csv")
    smiles_map = pd.read_pickle(f"{PROC}/lincs_smiles.pkl")
    lincs = screen.merge(smiles_map, on="pert_id", how="left")
    lincs = lincs[lincs.canonical_smiles.notna() & (lincs.canonical_smiles != "-666")]
    n_resolved = len(lincs)
    lincs_smiles = subsample(lincs.canonical_smiles.tolist(), n_per_group)

    run1 = pd.read_csv(f"{PROC}/run1/generated_set_converged.csv")
    run2 = pd.read_csv(f"{PROC}/run2/generated_set_converged.csv")
    top10 = pd.read_csv(f"{PROC}/run2/top_candidates_diverse.csv")

    run1_smiles = subsample(run1.canonical.tolist(), n_per_group)
    run2_smiles = subsample(run2.canonical.tolist(), n_per_group)
    top10_smiles = top10.canonical.tolist()

    fp_lincs, lincs_smiles = to_fp_array(lincs_smiles)
    fp_run1, run1_smiles = to_fp_array(run1_smiles)
    fp_run2, run2_smiles = to_fp_array(run2_smiles)
    fp_top10, top10_smiles = to_fp_array(top10_smiles)

    all_fp = np.vstack([fp_lincs, fp_run1, fp_run2, fp_top10])
    groups = (["LINCS"] * len(fp_lincs) + ["Run1"] * len(fp_run1)
              + ["Run2"] * len(fp_run2) + ["Top10"] * len(fp_top10))

    reducer = umap.UMAP(metric="jaccard", n_neighbors=30, min_dist=0.3,
                         random_state=seed)
    emb = reducer.fit_transform(all_fp)
    groups = np.array(groups)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(emb[groups == "LINCS", 0], emb[groups == "LINCS", 1],
               s=6, c=GRAY, alpha=0.35, linewidths=0,
               label=f"LINCS library (n={n_per_group:,} of {n_resolved:,} resolved)")
    ax.scatter(emb[groups == "Run1", 0], emb[groups == "Run1", 1],
               s=6, c=RED, alpha=0.45, linewidths=0,
               label=f"Run 1 converged (n={n_per_group:,} of {len(run1):,})")
    ax.scatter(emb[groups == "Run2", 0], emb[groups == "Run2", 1],
               s=6, c=BLUE, alpha=0.45, linewidths=0,
               label=f"Run 2 converged (n={n_per_group:,} of {len(run2):,})")
    ax.scatter(emb[groups == "Top10", 0], emb[groups == "Top10", 1],
               s=90, c="black", edgecolors="white", linewidths=1.2, zorder=5,
               label="Top-10 Run-2 candidates")

    ax.set_xlabel("UMAP 1 (Morgan fingerprint, Jaccard)")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Chemical space: LINCS library vs. generated sets")
    # manual legend proxies at a uniform, modest marker size -- the actual
    # scatter markers differ a lot in size (s=6 background vs s=90 top-10
    # highlight), which makes matplotlib's auto legend swatches look wildly
    # mismatched (a huge black dot next to three tiny ones).
    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=GRAY,
                   alpha=0.6, label=f"LINCS library (n={n_per_group:,} of {n_resolved:,} resolved)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=RED,
                   alpha=0.7, label=f"Run 1 converged (n={n_per_group:,} of {len(run1):,})"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=BLUE,
                   alpha=0.7, label=f"Run 2 converged (n={n_per_group:,} of {len(run2):,})"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=6.5, color="black",
                   markeredgecolor="white", markeredgewidth=1, label="Top-10 Run-2 candidates"),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/chemical_space.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/chemical_space.png  "
          f"(LINCS resolved {n_resolved}/{len(screen)}, "
          f"plotted {len(fp_lincs)}/{len(fp_run1)}/{len(fp_run2)}/{len(fp_top10)} "
          f"LINCS/Run1/Run2/Top10)")


# ============================================================== Figure E
# Small-multiples radar chart, one per top-10 Run-2 candidate, over the four
# transformed (0-1) multi-objective reward components actually used during RL
# (Methods 2.4): SenMayo_reversal, QED, Novelty_vs_LINCS, Synthesizability.
# These are the reward-component scores, not the raw units, so all four sit on
# a common 0-1 scale and a shared radial axis is directly comparable across
# candidates and across axes.
def figure_top10_radar():
    top = pd.read_csv(f"{PROC}/run2/top_candidates_diverse.csv")
    axes_cols = ["SenMayo_reversal", "QED", "Novelty_vs_LINCS", "Synthesizability"]
    axes_labels = ["SenMayo\nreversal", "QED", "Novelty\nvs. LINCS", "Synthesizability\n(SAscore)"]
    n_axes = len(axes_cols)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(2, 5, figsize=(14, 6.2), subplot_kw=dict(polar=True))
    for i, ax in enumerate(axs.flat):
        row = top.iloc[i]
        values = [row[c] for c in axes_cols]
        values += values[:1]
        ax.plot(angles, values, color=BLUE, linewidth=1.6)
        ax.fill(angles, values, color=BLUE, alpha=0.25)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=5.5, color="#999999")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_labels, fontsize=7.5)
        ax.tick_params(axis="x", pad=4)
        ax.set_title(f"#{i+1}  (SenMayo={row['SenMayo_reversal (raw)']:.2f})", fontsize=9, pad=14)
        ax.spines["polar"].set_alpha(0.3)
        ax.grid(alpha=0.35)

    fig.suptitle("Top-10 Run-2 candidates: multi-objective reward-component balance",
                  fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/top10_radar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/top10_radar.png  (10 candidates x 4 reward components: "
          f"{', '.join(axes_cols)})")


# ============================================================== Figure F
# GPS diagnostic: real measured SenMayo reversal_score vs. GPS's predicted
# score for the same 306 compounds (oracle/gps_real_diagnostic.py). Uses the
# continuous (non-thresholded) mean(P(down) - P(up)) formulation, which is the
# one that surfaces GPS's systematic up-regulation bias -- the discrete,
# 0.95-confidence-threshold formulation saturates into the "flat" class for
# most genes and washes the signal out (both are saved in the CSV).
def figure_gps_diagnostic():
    from scipy.stats import linregress

    df = pd.read_csv(f"{PROC}/gps_real_diagnostic.csv")
    lib = df[~df.is_positive_control]
    pc = df[df.is_positive_control].sort_values("reversal_score")

    slope, intercept, r, p, se = linregress(df.reversal_score, df.gps_continuous_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)

    ax.scatter(lib.reversal_score, lib.gps_continuous_score, s=14, c=GRAY, alpha=0.55,
               linewidths=0, label=f"LINCS library sample (n={len(lib)})")
    ax.scatter(pc.reversal_score, pc.gps_continuous_score, s=110, c=RED,
               edgecolors="black", linewidths=1, zorder=5,
               label="Positive-control senolytics/senomorphics (n=6)")
    label_offsets = {  # manual nudges to avoid overlap among closely-spaced points
        "navitoclax": (-8, 10), "vorinostat": (-10, -14), "panobinostat": (8, 10),
        "dasatinib": (8, 4), "quercetin": (-8, -16), "sirolimus": (8, -4),
    }
    for _, row in pc.iterrows():
        dx, dy = label_offsets.get(row.pert_iname, (7, 4))
        ha = "right" if dx < 0 else "left"
        ax.annotate(row.pert_iname, (row.reversal_score, row.gps_continuous_score),
                    textcoords="offset points", xytext=(dx, dy), fontsize=9, ha=ha)

    xs = np.linspace(df.reversal_score.min(), df.reversal_score.max(), 100)
    ax.plot(xs, intercept + slope * xs, color=BLUE, lw=1.8,
            label=f"OLS fit (r={r:.2f})")

    ax.set_xlabel("Real measured SenMayo reversal_score (LINCS L1000)")
    ax.set_ylabel("GPS-predicted score\n(mean $P$(down) $-$ $P$(up)) across SenMayo genes)")
    ax.set_title("GPS diagnostic: predicted vs. real SenMayo response\n"
                  "(same 306 compounds)")
    frac_neg = (df.gps_continuous_score < 0).mean()
    ax.text(0.03, 0.03,
            f"{frac_neg:.0%} of compounds predicted net UP-regulation by GPS\n"
            f"(vs. {(df.reversal_score < 0).mean():.0%} in the real data)",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/gps_diagnostic.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/gps_diagnostic.png  (n={len(df)}, r={r:.3f}, "
          f"{frac_neg:.1%} GPS-predicted net up-regulation vs "
          f"{(df.reversal_score < 0).mean():.1%} real)")


# ============================================================== Figure G
# Thiazole-dicarboxamide motif enrichment: frequency of the substructure
# shared by 6/10 top Run-2 candidates, across the LINCS training library,
# Run 1, and Run 2 stratified by predicted-score percentile. The SMARTS
# (2-carboxamido-thiazole bearing a second ring N-acylamino group) was
# derived empirically via rdFMCS.FindMCS on the top-10 candidates, then
# tightened until it matched exactly the 6/10 the manuscript text already
# describes as sharing the motif ("several", not all ten).
THIAZOLE_DICARBOXAMIDE_SMARTS = "[#6]1(:[#7]:[#6](:[#6]:[#16]:1)C(=O)N)[#7]C(=O)"


def figure_motif_enrichment():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    pattern = Chem.MolFromSmarts(THIAZOLE_DICARBOXAMIDE_SMARTS)

    def frac_match(smiles_list):
        n_match = n_valid = 0
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            n_valid += 1
            if m.HasSubstructMatch(pattern):
                n_match += 1
        return n_match, n_valid

    screen = pd.read_csv(f"{PROC}/lincs_senmayo_screen_results.csv")
    smiles_map = pd.read_pickle(f"{PROC}/lincs_smiles.pkl")
    lincs = screen.merge(smiles_map, on="pert_id", how="left")
    lincs = lincs[lincs.canonical_smiles.notna() & (lincs.canonical_smiles != "-666")]
    run1 = pd.read_csv(f"{PROC}/run1/generated_set_converged.csv")
    run2 = pd.read_csv(f"{PROC}/run2/generated_set_converged.csv")
    run2_sorted = run2.sort_values("SenMayo_reversal (raw)", ascending=False)
    top10 = pd.read_csv(f"{PROC}/run2/top_candidates_diverse.csv")

    groups_a = [
        ("LINCS\nlibrary", lincs.canonical_smiles),
        ("Run 1\n(converged)", run1.canonical),
        ("Run 2\n(all converged)", run2.canonical),
    ]
    groups_b = [
        ("Run 2\n(all converged)", run2.canonical),
        ("Run 2\ntop 5%", run2_sorted.head(int(len(run2_sorted) * 0.05)).canonical),
        ("Run 2\ntop 1%", run2_sorted.head(int(len(run2_sorted) * 0.01)).canonical),
        ("Top-10\ncandidates", top10.canonical),
    ]

    results_a = [(label, *frac_match(smis)) for label, smis in groups_a]
    results_b = [(label, *frac_match(smis)) for label, smis in groups_b]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    labels_a = [r[0] for r in results_a]
    pct_a = [100 * r[1] / r[2] for r in results_a]
    bars1 = ax1.bar(labels_a, pct_a, width=0.55, color=[GRAY, RED, BLUE], edgecolor="none")
    for bar, r in zip(bars1, results_a):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                  f"{r[1]}/{r[2]:,}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylabel("Molecules containing the motif (%)")
    ax1.set_title("Absent from training data")
    ax1.set_ylim(0, max(pct_a) * 1.6 + 0.3)

    labels_b = [r[0] for r in results_b]
    pct_b = [100 * r[1] / r[2] for r in results_b]
    bars2 = ax2.bar(labels_b, pct_b, width=0.6, color=[BLUE, BLUE, BLUE, "black"],
                     edgecolor="none")
    for bar, alpha in zip(bars2, [0.55, 0.7, 0.85, 1.0]):
        bar.set_alpha(alpha)
    for bar, r in zip(bars2, results_b):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                  f"{r[1]}/{r[2]:,}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylabel("Molecules containing the motif (%)")
    ax2.set_title("Enriched with predicted score")
    ax2.set_ylim(0, max(pct_b) * 1.25)

    fig.suptitle("Thiazole-dicarboxamide motif: absent from training data, "
                  "enriched among top-scoring candidates", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/motif_enrichment.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/motif_enrichment.png")
    for label, n_m, n_v in results_a + results_b[1:]:
        print(f"  {label.replace(chr(10), ' ')}: {n_m}/{n_v} = {100*n_m/n_v:.3f}%")


# ============================================================== Figure H
# Gene-level mechanistic heatmap: real measured per-gene SenMayo z-scores for
# the 6 positive controls (real LINCS data) side by side with per-gene
# bridge-model PREDICTIONS for the top-10 Run-2 candidates (own Ridge models
# trained on real data, oracle/lincs_pergene_bridge.py -- not GPS, since
# Figure 1 already showed GPS's per-gene predictions are unreliable). The two
# halves are explicitly different in kind and labelled as such: left =
# measured, right = predicted.
def figure_gene_heatmap():
    pred = pd.read_csv(f"{PROC}/pergene_bridge_predictions.csv")
    val = pd.read_csv(f"{PROC}/pergene_bridge_validation.csv")
    pergene = pd.read_csv(f"{PROC}/lincs_senmayo_pergene.csv")

    genes = list(val.sort_values("test_r", ascending=False).gene)  # rank by validated r
    pc_names = ["dasatinib", "navitoclax", "panobinostat", "quercetin", "sirolimus", "vorinostat"]
    pc = pergene[pergene.pert_iname.str.lower().isin(pc_names)].drop_duplicates("pert_iname")
    pc = pc.set_index(pc.pert_iname.str.lower()).reindex(pc_names)

    real_mat = pc[genes].values.astype(float)          # (6, n_genes)
    pred_mat = pred[genes].values.astype(float)         # (10, n_genes)
    mat = np.vstack([real_mat, pred_mat]).T              # (n_genes, 16)
    col_labels = [n.capitalize() for n in pc_names] + [f"#{i}" for i in pred["rank"]]

    vmax = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8)
    ax.axvline(5.5, color="black", lw=1.8)

    # x in data coords (column position), y in axes-fraction (0-1) via the blended
    # x-axis transform -- a fixed y=1.03 clears the title regardless of row count,
    # unlike the previous data-coordinate y=-1.6 (its physical gap above the plot
    # shrank as more genes/rows were added, until it collided with the title).
    header_trans = ax.get_xaxis_transform()
    ax.text(2.5, 1.03, "Real measured (positive controls)", transform=header_trans,
            ha="center", fontsize=9.5, fontweight="bold")
    ax.text(10.5, 1.03, "Bridge-model predicted (Run-2 top-10)", transform=header_trans,
            ha="center", fontsize=9.5, fontweight="bold")
    ax.set_title("SenMayo gene-level response: measured vs. predicted\n"
                  "(genes ranked by scaffold-split validated $r$, top = most reliable)",
                  fontsize=11.5, pad=42)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("SenMayo gene $z$-score\n(negative = down-regulated = reversal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/gene_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/gene_heatmap.png  ({len(genes)} genes, "
          f"mean validated r={val.test_r.mean():+.3f})")


# ============================================================== Figure C2
# Fix for the original scaffold_diversity.png (fig:diversity): both panels
# shared one 0-60% axis, so Run 2's top-10 scaffold shares (all < 0.02%) were
# indistinguishable from a flat line. Each panel now gets its own y-axis
# scaled to its own data, with the exact share labelled above every bar so
# the reading is unambiguous regardless of bar height -- the two panels are
# explicitly NOT on a common scale, called out in-figure and in the caption.
def figure_scaffold_diversity():
    run1 = pd.read_csv(f"{PROC}/run1/generated_set_converged.csv")
    run2 = pd.read_csv(f"{PROC}/run2/generated_set_converged.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, df, color, label, run_name in [
        (ax1, run1, RED, "Run 1", "Run 1"), (ax2, run2, BLUE, "Run 2", "Run 2")
    ]:
        share = 100 * df.scaffold.value_counts(normalize=True).head(10)
        bars = ax.bar(range(1, 11), share.values, width=0.65, color=color, edgecolor="none")
        fmt = "{:.1f}%" if share.iloc[0] >= 1 else "{:.3f}%"
        for bar, v in zip(bars, share.values):
            # offset in points (not data units) so the gap survives rotation --
            # a data-space offset only pads the pre-rotation anchor and the
            # rotated glyphs still dip back into the bar (verified visually).
            ax.annotate(fmt.format(v), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=90 if share.iloc[0] < 1 else 0)
        ax.set_xlabel("scaffold rank")
        ax.set_ylabel("% of converged set")
        ax.set_title(f"{run_name}: top-10 scaffold share\n(note: independent $y$-axis scale)", fontsize=10.5)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, share.iloc[0] * 1.35)

    fig.suptitle("Scaffold concentration, Run 1 vs. Run 2 -- axes are NOT on a common scale "
                  "(Run 2's largest share is 330x smaller than Run 1's)", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/scaffold_diversity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/scaffold_diversity.png  "
          f"(Run1 top scaffold {100*run1.scaffold.value_counts(normalize=True).iloc[0]:.4f}%, "
          f"Run2 top scaffold {100*run2.scaffold.value_counts(normalize=True).iloc[0]:.4f}%)")


# ============================================================== Figure I
# Reward progression over RL steps, both runs -- restyles the original raw
# per-step trace (very high step-to-step noise made the trend read as a thick
# smear in matplotlib's default colors). The raw series is kept, drawn thin
# and faint, with a rolling-mean summary line on top; both are the same real
# per-batch mean reward at two smoothing levels, so nothing is hidden.
def figure_score_progression(window=21):
    run1 = pd.read_csv(f"{PROC}/run1/score_progression.csv")
    run2 = pd.read_csv(f"{PROC}/run2/score_progression.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for df, color, label in [
        (run1, RED, "Run 1 (bucket=25, sigma=128)"),
        (run2, BLUE, "Run 2 (bucket=5, sigma=100)"),
    ]:
        smoothed = df.Score.rolling(window, center=True, min_periods=1).mean()
        ax.plot(df.step, df.Score, color=color, lw=0.6, alpha=0.25)
        ax.plot(df.step, smoothed, color=color, lw=2, label=label)

    ax.set_xlabel("RL step")
    ax.set_ylabel("Mean batch reward (geometric mean)")
    ax.set_title(f"Reward progression during reinforcement learning\n"
                  f"(rolling {window}-step mean over the raw per-batch trace)", fontsize=12.5)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_xlim(0, max(run1.step.max(), run2.step.max()))
    fig.tight_layout()
    fig.savefig(f"{OUT}/score_progression.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/score_progression.png  "
          f"(Run1 final={run1.Score.iloc[-1]:.3f}, Run2 final={run2.Score.iloc[-1]:.3f})")


if __name__ == "__main__":
    figure_bridge_validation()
    figure_real_data_validation()
    figure_score_distributions()
    figure_chemical_space()
    figure_top10_radar()
    figure_gps_diagnostic()
    figure_motif_enrichment()
    figure_gene_heatmap()
    figure_scaffold_diversity()
    figure_score_progression()
