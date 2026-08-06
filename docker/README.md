# Docker image

A self-contained image with REINVENT4, the SenMayo/SASP-reversal bridge model, and
the trained Run-2 agent -- built for other researchers to reuse the pipeline without
retraining anything.

## Get the image

### Pull (recommended)

No build needed:

```
docker pull ghcr.io/dralperenuysal/reinvent-geroprotector:latest
```

`latest` tracks ongoing updates. To pull the exact image used to produce the
results reported in the manuscript, pull the pinned tag instead:

```
docker pull ghcr.io/dralperenuysal/reinvent-geroprotector:v1.0.0
```

The usage examples below use the `latest` image name; if you pulled `v1.0.0` or
built locally instead, substitute the tag/name accordingly.

### Build from source

From the project root (needs `data/processed/bridge_model.joblib` and
`reinvent_run/outputs/run2_stage1.chkpt`, both present in a full checkout):

```
docker build -f docker/Dockerfile -t reinvent-geroprotector .
```

Build time is a few minutes; the image is approximately 3.5GB (mostly PyTorch,
REINVENT4's dependencies, and the bundled model/prior weights). Requires network
access during build: REINVENT4 is cloned fresh from its
[canonical repository](https://github.com/MolecularAI/REINVENT4), and its prior
model weights are downloaded from REINVENT4's
[Zenodo deposit](https://doi.org/10.5281/zenodo.15641296) (pinned to a specific
version record for reproducibility).

## Usage

All three subcommands read/write through a mounted `/data` directory.

### Score your own compounds against the trained oracle

No REINVENT4 involved -- loads `bridge_model.joblib` directly and predicts
`reversal_score` from structure. Input is a CSV/TSV with a `smiles` column, or a
plain text file with one SMILES per line.

```
docker run --rm -v $(pwd):/data ghcr.io/dralperenuysal/reinvent-geroprotector:latest score \
    --input /data/my_compounds.csv --output /data/scores.csv
```

### Sample novel molecules from the trained Run-2 agent

Fast (seconds to minutes on CPU); samples directly from the post-RL agent reported
in the manuscript (diverse, non-collapsed run), no training involved.

```
docker run --rm -v $(pwd):/data ghcr.io/dralperenuysal/reinvent-geroprotector:latest sample \
    --config /opt/senmayo/configs/sample_run2_agent.toml --log /data/sample.log
```

Edit `num_smiles` in `docker/configs/sample_run2_agent.toml` (rebuild required) or
mount a modified copy at a different path and point `--config` at it.

### Run reinforcement learning from scratch

Reproduces the actual Run-2 training (3,000 steps; CPU by default, edit `device` in
the config for GPU). This is the compute-heavy path -- expect hours, not minutes.

```
docker run --rm -v $(pwd):/data ghcr.io/dralperenuysal/reinvent-geroprotector:latest generate \
    --config /opt/senmayo/configs/staged_learning_run2.toml --log /data/run.log
```

Outputs (`run2_1.csv`, checkpoints, TensorBoard logs) land in the mounted `/data`
directory per the config's `summary_csv_prefix` / `tb_logdir` settings.

### Swap in your own oracle

`docker/comp_senmayo_bridge.py` is REINVENT4's plugin API applied to a `.joblib`
model exposing `.predict()` on the same Morgan-fingerprint + physicochemical-descriptor
feature layout (see `oracle/lincs_structure_bridge.py`'s `featurize()`). Mount a
different `.joblib` and edit `params.model_file` in a copy of the staged-learning
config to point at it, no other pipeline changes needed -- this is the modularity the
manuscript's Methods 2.3 refers to.

### Interactive shell

```
docker run --rm -it -v $(pwd):/data --entrypoint bash ghcr.io/dralperenuysal/reinvent-geroprotector:latest
```

## Notes on how this was built

- REINVENT4 is `pip install`-ed (not editable), which copies `reinvent_plugins`
  into site-packages as its own top-level package; the custom `SenmayoBridge`
  component is therefore installed into the actual site-packages location, not the
  `/opt/REINVENT4` source tree, resolved dynamically at build time
  (`reinvent_plugins.components.__path__`) rather than a hardcoded Python version
  path.
- The original `SenmayoBridge` component source file used for the paper's actual
  runs was not preserved in the repository. `docker/comp_senmayo_bridge.py` is a
  faithful reconstruction from the trained model artifact (which stores the exact
  featurization parameters: fingerprint radius, bit count, descriptor
  mean/SD) and REINVENT4's own plugin API (pattern taken from its built-in
  `TanimotoSimilarity` component). Verified against historical outputs: predictions
  for dasatinib, aspirin, and the manuscript's top-1 candidate reproduce the
  original recorded values to within floating-point noise.
- `libxrender1`/`libxext6`/`libsm6` are required by RDKit's `Draw` module (imported
  transitively by REINVENT4's reporting code) on a headless Debian base image;
  omitting them fails with `ImportError: libXrender.so.1: cannot open shared object
  file`.
- A `scikit-learn` `InconsistentVersionWarning` on model load (pickled with 1.8.0,
  loaded with a newer 1.x) is expected and harmless -- verified by reproducing known
  historical predictions exactly.
