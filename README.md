# Generative Molecular Design for SASP Reversal

Generative molecular design guided by a reward oracle trained on real LINCS L1000
transcriptomic data, applied to reversing the senescence-associated secretory
phenotype (SASP) via [REINVENT4](https://github.com/MolecularAI/REINVENT4).

Cellular senescence contributes to aging and age-related disease largely through the
SASP, a secreted program of pro-inflammatory cytokines, chemokines, and proteases.
SenMayo is a validated 125-gene transcriptomic signature of SASP activity. This
project trains a structure-based oracle that predicts a compound's effect on the
SenMayo signature directly from real, measured LINCS L1000 perturbational
transcriptomics (not a pretrained black-box model), then uses that oracle as the
reward function for a REINVENT4 reinforcement-learning run, producing novel,
drug-like candidate molecules predicted to reverse SASP.

## Key results

- Reward oracle (Random Forest on Morgan fingerprints + physicochemical descriptors)
  trained on 21,299 real LINCS L1000 compounds, scaffold-split Pearson r = 0.263
  (permutation null r = -0.000 +/- 0.013).
- A pretrained alternative (GPS, a published transcriptomic-effect predictor) was
  evaluated first and rejected: a targeted diagnostic against real measured LINCS
  data showed a systematic bias toward predicting SASP up-regulation for almost any
  compound (88.6% of 306 tested compounds, vs. 55.9% in the real data), including all
  six known senolytics/senomorphics used as positive controls. See
  `oracle/gps_real_diagnostic.py`.
- First reinforcement-learning run collapsed onto a single trivial Murcko scaffold
  (57.2% of the converged set) despite an active diversity filter. A second run, with
  a tightened diversity filter and lower reward sharpness, eliminated the collapse
  (88.4% distinct scaffolds among 92,454 converged molecules) while also improving
  the mean predicted SASP-reversal score by 0.61 library standard deviations over the
  training library average.
- The reward function learned a preference for a thiazole-dicarboxamide substructure
  that is entirely absent from the training library (0/21,220 compounds) but present
  in 24.1% of the top 1% of generated molecules by predicted score, consistent with
  genuine de novo discovery rather than memorization.

## Repository layout

```
oracle/           Reward-oracle pipeline: real-data SenMayo screen, structure-based
                   bridge model (the deployed REINVENT4 oracle), per-gene bridge
                   models, and the GPS evaluation/rejection diagnostics
reinvent_run/      REINVENT4 configs (staged learning, scoring, sampling) and run
                   outputs/logs
analysis/          Post-hoc analysis of generated sets (scaffold diversity, top
                   candidates) and manuscript figure generation
data/              Raw (gitignored) and processed data
gps/               GPS model clone, used only for the rejection diagnostic
                   (gitignored; see oracle/gps_real_diagnostic.py)
paper/             Manuscript, conference abstract, and figures (gitignored, kept
                   local; see Data and code availability below)
docker/            Self-contained Docker image (Dockerfile, CLI, run configs);
                   see Docker below
```

## Pipeline

1. **Real-data SenMayo reversal screen** (`oracle/lincs_senmayo_screen.py`) — scores
   every LINCS L1000 compound signature by its mean z-score across the SenMayo genes,
   standardized against a permutation null, yielding a per-compound `reversal_score`.
2. **Structure-to-reversal-score bridge model** (`oracle/lincs_structure_bridge.py`) —
   trains Ridge and Random Forest regressors mapping molecular structure (Morgan
   fingerprints + physicochemical descriptors) to `reversal_score` under a
   Bemis-Murcko scaffold split; the better model is refit on the full library and
   deployed as the REINVENT4 oracle.
3. **REINVENT4 reinforcement learning** (`reinvent_run/`) — a custom scoring component
   wraps the bridge model and combines it with QED, novelty-vs-library, and synthetic
   accessibility into a multi-objective reward. Run twice under different
   diversity-filter settings (see `reinvent_run/configs/`).
4. **Post-hoc analysis** (`analysis/`) — scaffold diversity, top-candidate selection,
   and all figures reported in the manuscript.
5. **GPS evaluation and rejection** (`oracle/gps_*.py`) — an earlier, alternative
   oracle design based on the pretrained GPS model, kept in the repository as a
   documented negative result rather than deleted.

Two supplementary per-gene models (`oracle/lincs_senmayo_pergene.py`,
`oracle/lincs_pergene_bridge.py`) extend the aggregate oracle to the 24 individual
SenMayo genes most correlated with `reversal_score`, for a gene-level mechanistic
view of the top candidates alongside real measured data for known senolytics.

## Docker

A self-contained image (REINVENT4 + the trained bridge oracle + the trained Run-2
agent) lets other researchers score compounds, sample new candidates, or run the
full reinforcement-learning pipeline without installing anything locally:

```
docker pull ghcr.io/dralperenuysal/reinvent-geroprotector:latest
docker run --rm -v $(pwd):/data ghcr.io/dralperenuysal/reinvent-geroprotector:latest score \
    --input /data/my_compounds.csv --output /data/scores.csv
```

See `docker/README.md` for the full usage guide (sampling from the trained agent,
running RL from scratch, swapping in a different oracle, and building from source
instead of pulling).

## Data and code availability

Raw LINCS L1000 data, the SenMayo gene list, and REINVENT4 run artifacts are
gitignored (tens of GB, reproducible from public sources: LINCS L1000 via the Gene
Expression Omnibus, accessions GSE70138 and GSE92742; the SenMayo gene list from
Supplementary Data 1 of Saul et al. 2022, *Nature Communications*). Small processed
results (screen results, validation metrics, held-out predictions) are tracked in
`data/processed/`; the trained bridge model itself (`bridge_model.joblib`, ~243MB) is
excluded for size but is reproducible by rerunning `oracle/lincs_structure_bridge.py`
against the raw data.

## Requirements

Python with `rdkit`, `torch`, `scikit-learn`, `pandas`, `numpy`, `scipy`,
`matplotlib`, `umap-learn`, `h5py`, and `joblib` -- pinned to the versions used to
produce the manuscript's results in `requirements.txt`:

```
pip install torch==2.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

REINVENT4 itself is a separate install (see its own
[repository](https://github.com/MolecularAI/REINVENT4)); the pinned versions above
cover the oracle/analysis pipeline (`oracle/`, `analysis/`) only.

## Limitations

LINCS L1000 was measured predominantly in immortalized cancer cell lines; whether the
structure-to-transcriptome relationship learned here transfers to primary or
senescent cells has not been experimentally verified. The bridge model's scaffold-split
r = 0.26 sets a modest, honest ceiling on oracle accuracy. Synthetic accessibility is
scored with a fragment-complexity heuristic (SAScore), not a real synthesis-route
planner. No wet-lab validation has been performed; all reported scores are model
predictions.

## License

MIT (see `LICENSE`). REINVENT4 itself is a separate dependency under its own
Apache-2.0 license (cloned fresh at Docker build time, not vendored in this
repository); see its [repository](https://github.com/MolecularAI/REINVENT4) for
terms.

## Author

S. Alperen Uysal, MD, PhD — Ege University School of Medicine, Department of Medical
Biology and Genetics. dralperenuysal@gmail.com
