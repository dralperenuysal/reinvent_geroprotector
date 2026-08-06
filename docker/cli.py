#!/usr/bin/env python
"""CLI for the reinvent_geroprotector Docker image.

Subcommands:
  score      Score a file of SMILES against the trained SenMayo reversal-score
             bridge model directly (no REINVENT4 involved).
  generate   Run REINVENT4 (staged learning / reinforcement learning) with the
             SenMayo bridge oracle wired in as the reward.
  sample     Run REINVENT4 sampling from an already-trained agent checkpoint
             (fast; does not retrain).

Examples:
  docker run -v $(pwd):/data reinvent-geroprotector score \\
      --input /data/my_compounds.csv --output /data/scores.csv

  docker run -v $(pwd):/data reinvent-geroprotector generate \\
      --config /opt/senmayo/configs/staged_learning_run2.toml --log /data/run.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

DESCRIPTOR_FUNCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.RingCount,
    Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.NumAromaticRings,
]

DEFAULT_MODEL = "/opt/senmayo/bridge_model.joblib"


def featurize(smilies, state):
    gen = AllChem.GetMorganGenerator(radius=state["radius"], fpSize=state["n_bits"])
    rows, valid = [], []
    for smi in smilies:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid.append(False)
            rows.append(np.zeros(state["n_bits"] + len(DESCRIPTOR_FUNCS), dtype=np.float32))
            continue
        valid.append(True)
        fp = np.zeros(state["n_bits"], dtype=np.float32)
        for i, c in gen.GetCountFingerprint(mol).GetNonzeroElements().items():
            fp[i] = c
        d = np.array([f(mol) for f in DESCRIPTOR_FUNCS], dtype=np.float32)
        rows.append(np.concatenate([np.log1p(fp), (d - state["desc_mu"]) / state["desc_sd"]]))
    return np.vstack(rows), np.array(valid)


def cmd_score(args):
    state = joblib.load(args.model)
    print(f"Loaded bridge model: {state.get('validation', {}).get('model', 'unknown')} "
          f"(scaffold-split r={state.get('validation', {}).get('test_r', float('nan')):+.3f}, "
          f"trained on {state.get('n_train', '?')} compounds)", file=sys.stderr)

    if args.input.endswith((".csv", ".tsv")):
        sep = "\t" if args.input.endswith(".tsv") else ","
        df = pd.read_csv(args.input, sep=sep)
        col = next((c for c in df.columns if c.lower() == "smiles"), None)
        if col is None:
            sys.exit(f"error: no 'smiles' column found in {args.input} (columns: {list(df.columns)})")
        smilies = df[col].astype(str).tolist()
    else:
        with open(args.input) as f:
            smilies = [l.strip() for l in f if l.strip()]
        df = pd.DataFrame({"smiles": smilies})

    X, valid = featurize(smilies, state)
    preds = state["model"].predict(X)
    df["reversal_score_predicted"] = np.where(valid, preds, np.nan)
    df["valid_smiles"] = valid
    n_invalid = (~valid).sum()
    if n_invalid:
        print(f"warning: {n_invalid}/{len(smilies)} SMILES could not be parsed by RDKit "
              f"(reversal_score_predicted = NaN for those rows)", file=sys.stderr)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} scored compounds to {args.output}", file=sys.stderr)


def cmd_generate(args):
    cmd = ["reinvent"]
    if args.log:
        cmd += ["-l", args.log]
    cmd += [args.config]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def cmd_sample(args):
    cmd = ["reinvent"]
    if args.log:
        cmd += ["-l", args.log]
    cmd += [args.config]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score SMILES against the trained bridge model")
    p_score.add_argument("--input", required=True, help="CSV/TSV with a 'smiles' column, or one SMILES per line")
    p_score.add_argument("--output", required=True, help="Output CSV path")
    p_score.add_argument("--model", default=DEFAULT_MODEL)
    p_score.set_defaults(func=cmd_score)

    p_gen = sub.add_parser("generate", help="Run REINVENT4 staged learning (reinforcement learning)")
    p_gen.add_argument("--config", required=True, help="Path to a staged_learning TOML config")
    p_gen.add_argument("--log", default=None)
    p_gen.set_defaults(func=cmd_generate)

    p_sam = sub.add_parser("sample", help="Sample molecules from a trained agent checkpoint")
    p_sam.add_argument("--config", required=True, help="Path to a sampling TOML config")
    p_sam.add_argument("--log", default=None)
    p_sam.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
