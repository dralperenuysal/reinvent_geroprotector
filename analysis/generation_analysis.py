#!/usr/bin/env python
"""AGENTS.md Section 8 deliverables: generation statistics, novelty, comparison
against the LINCS library, scaffold diversity (mode-collapse check), and top
candidates for the abstract figure.

"The generated set" = unique, valid molecules from the last quarter of the RL
run (the converged agent, not the early exploratory/near-prior samples).

Usage: generation_analysis.py <run_name> <csv_path>
  run1: reinvent_run/outputs/run1_archive/production_1.csv (archived)
  run2: reinvent_run/outputs/run2_1.csv
Top candidates are capped at one per Murcko scaffold, so the reported set is
genuinely diverse rather than N near-identical analogs of one motif -- run 1's
naive top-N-by-score selection was all one scaffold (see run1's analysis).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = f"{ROOT}/reinvent_run"
CONVERGED_FRACTION = 0.25
N_TOP = 10


def analyze(run_name: str, csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== SECTION 8: GENERATION ANALYSIS -- {run_name} ===\n")

    df = pd.read_csv(csv_path)
    n_total = len(df)
    n_steps = df.step.max()

    # ---------------------------------------------------------- overall generation stats
    n_valid = (df.SMILES_state != 0).sum()
    n_new_valid = (df.SMILES_state == 1).sum()
    print(f"--- 1. Generation statistics ({n_steps} steps) ---")
    print(f"Total samples: {n_total:,}")
    print(f"Valid SMILES: {n_valid:,} ({100*n_valid/n_total:.2f}%)")
    print(f"Novel valid (not a repeat within the run): {n_new_valid:,} ({100*n_new_valid/n_total:.2f}%)")

    def canonicalize_closed_shell(s):
        """RDKit's SMILES parser accepts explicit-H bracket atoms like '[CH]' even when
        the resulting valence is a free radical (e.g. a 3-bond carbon), which is not a
        viable drug candidate -- REINVENT4's own validity check is just RDKit-parseable,
        so radicals slip through as 'valid' unless filtered here explicitly."""
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):
            return None
        return Chem.MolToSmiles(m)

    valid_df = df[df.SMILES_state != 0].copy()
    valid_df["canonical"] = valid_df.SMILES.apply(canonicalize_closed_shell)
    n_radical = valid_df.SMILES.apply(
        lambda s: Chem.MolFromSmiles(s) is not None
    ).sum() - valid_df.canonical.notna().sum()
    valid_df = valid_df.dropna(subset=["canonical"])
    n_unique_total = valid_df.canonical.nunique()
    print(f"Radical-electron species excluded (RDKit-valid but not closed-shell): {n_radical:,}")
    print(f"Unique valid molecules across the whole run: {n_unique_total:,}")

    score_by_step = df.groupby("step")["Score"].mean()
    print(f"Mean score: step 1 = {score_by_step.iloc[0]:.3f}, "
          f"step {n_steps} = {score_by_step.iloc[-1]:.3f}, "
          f"peak = {score_by_step.max():.3f} (step {score_by_step.idxmax()})")
    score_by_step.to_csv(f"{out_dir}/score_progression.csv")

    # ---------------------------------------------------------- converged generated set
    cutoff = int(n_steps * (1 - CONVERGED_FRACTION))
    converged = valid_df[(valid_df.step > cutoff) & (valid_df.SMILES_state == 1)].copy()
    converged = converged.drop_duplicates(subset="canonical")
    print(f"\n--- Converged generated set (steps {cutoff+1}-{n_steps}) ---")
    print(f"Unique novel valid molecules: {len(converged):,}")

    converged["scaffold"] = converged.canonical.apply(
        lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "__none__"
    )
    n_scaffolds = converged.scaffold.nunique()
    top_scaffold_frac = converged.scaffold.value_counts(normalize=True).iloc[0]
    print(f"Distinct Murcko scaffolds: {n_scaffolds:,} ({n_scaffolds/len(converged)*100:.1f}% of molecules)")
    print(f"Most common single scaffold: {top_scaffold_frac*100:.1f}% of the converged set "
          f"{'<-- mode collapse risk' if top_scaffold_frac > 0.1 else '(diverse)'}")
    print("Top 5 scaffolds by share:")
    print((converged.scaffold.value_counts(normalize=True).head(5) * 100).round(1))

    # ---------------------------------------------------------- comparison vs LINCS library
    screen = pd.read_csv(f"{ROOT}/data/processed/lincs_senmayo_screen_results.csv")
    lib_mean = screen.reversal_score.mean()
    lib_std = screen.reversal_score.std()
    gen_mean = converged["SenMayo_reversal (raw)"].mean()
    gen_std = converged["SenMayo_reversal (raw)"].std()
    print(f"\n--- 4. Comparison: generated set vs LINCS library ---")
    print(f"LINCS library (n={len(screen):,}): mean = {lib_mean:+.4f}, std = {lib_std:.4f}")
    print(f"Generated set (n={len(converged):,}): mean = {gen_mean:+.4f}, std = {gen_std:.4f}")
    print(f"Improvement: {gen_mean - lib_mean:+.4f} ({(gen_mean-lib_mean)/lib_std:.2f} library SDs)")

    # ---------------------------------------------------------- novelty vs LINCS
    print(f"\n--- 3. Novelty: nearest-neighbor Tanimoto similarity to the LINCS library ---")
    lincs_smiles = pd.read_pickle(f"{ROOT}/data/raw/lincs/smiles.pkl")
    lincs_valid = lincs_smiles[lincs_smiles.canonical_smiles != "-666"].canonical_smiles.dropna().unique()
    rng = np.random.RandomState(1)
    lincs_sample = rng.choice(lincs_valid, size=min(3000, len(lincs_valid)), replace=False)

    gen_mol = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
    lincs_fps = [gen_mol.GetFingerprint(m) for m in
                 (Chem.MolFromSmiles(s) for s in lincs_sample) if m is not None]

    nn_sims = []
    for s in converged.canonical:
        fp = gen_mol.GetFingerprint(Chem.MolFromSmiles(s))
        nn_sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, lincs_fps)))
    converged["nn_similarity_to_lincs"] = nn_sims

    print(f"Nearest-neighbor Tanimoto similarity to LINCS library (n={len(lincs_fps):,} sampled):")
    print(f"  mean = {np.mean(nn_sims):.3f}, median = {np.median(nn_sims):.3f}, max = {np.max(nn_sims):.3f}")
    print(f"  molecules with nn_similarity > 0.85 (near-copies): "
          f"{(np.array(nn_sims) > 0.85).sum()}/{len(nn_sims)}")

    # ---------------------------------------------------------- top candidates (diverse: 1/scaffold)
    print(f"\n--- 5. Top {N_TOP} candidates by SenMayo_reversal (raw), one per scaffold ---")
    ranked = converged.sort_values("SenMayo_reversal (raw)", ascending=False)
    top = ranked.drop_duplicates(subset="scaffold").head(N_TOP).reset_index(drop=True)
    show_cols = ["canonical", "SenMayo_reversal (raw)", "QED (raw)", "Synthesizability (raw)",
                 "nn_similarity_to_lincs", "scaffold"]
    print(top[show_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    top.to_csv(f"{out_dir}/top_candidates_diverse.csv", index=False)
    converged.to_csv(f"{out_dir}/generated_set_converged.csv", index=False)
    print(f"\nSaved {out_dir}/top_candidates_diverse.csv and generated_set_converged.csv")

    mols = [Chem.MolFromSmiles(s) for s in top.canonical]
    legends = [f"#{i+1}  score={r:.2f}  QED={q:.2f}"
               for i, (r, q) in enumerate(zip(top["SenMayo_reversal (raw)"], top["QED (raw)"]))]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=legends)
    img.save(f"{out_dir}/top_candidates_diverse.png")
    print(f"Saved 2D depiction to {out_dir}/top_candidates_diverse.png")

    return dict(run=run_name, n_scaffolds=n_scaffolds, n_converged=len(converged),
                top_scaffold_frac=top_scaffold_frac, gen_mean=gen_mean, gen_std=gen_std,
                lib_mean=lib_mean, nn_sim_mean=np.mean(nn_sims), n_valid=n_valid, n_total=n_total,
                final_score=score_by_step.iloc[-1])


if __name__ == "__main__":
    run_name, csv_path = sys.argv[1], sys.argv[2]
    out_dir = f"{ROOT}/data/processed/{run_name}"
    analyze(run_name, csv_path, out_dir)
