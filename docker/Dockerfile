# reinvent-geroprotector: REINVENT4 + a SenMayo/SASP-reversal reward oracle
# trained on real LINCS L1000 data. Build from the project root:
#
#   docker build -f docker/Dockerfile -t reinvent-geroprotector .
#
# See docker/README.md for usage.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates curl \
        libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# --- REINVENT4, cloned fresh from its canonical public repo (not vendored) ---
RUN git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git /opt/REINVENT4
WORKDIR /opt/REINVENT4
RUN pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu

# Prior model weights: REINVENT4's own repo no longer carries these in git
# (moved out upstream); the project's README now points to Zenodo
# (concept DOI 10.5281/zenodo.15641296). Pinned to a specific, permanent
# Zenodo version record (20701824) rather than re-resolving "latest" on every
# build, so this Dockerfile keeps working even if a newer prior set is
# published later.
RUN mkdir -p /opt/REINVENT4/priors && \
    curl -sL -o /opt/REINVENT4/priors/reinvent_pubchem.prior \
        "https://zenodo.org/api/records/20701824/files/reinvent_pubchem.prior/content" && \
    test $(stat -c%s /opt/REINVENT4/priors/reinvent_pubchem.prior) -gt 20000000

# Extra dependencies for the SenMayo bridge oracle (not required by REINVENT4 itself)
RUN pip install --no-cache-dir scikit-learn joblib

# --- Custom scoring component: SenMayo reversal-score bridge model ---
# `pip install .` copies reinvent_plugins into site-packages as its own
# top-level package (REINVENT4's component discovery walks that installed
# copy at runtime via pkgutil, not the /opt/REINVENT4 source tree), so the
# new component has to be dropped into the actual installed location --
# resolved dynamically rather than hardcoding a python3.11 site-packages path.
RUN COMPONENTS_DIR=$(cd /tmp && python -c "import reinvent_plugins.components as c; print(c.__path__[0])") && \
    case "$COMPONENTS_DIR" in /opt/REINVENT4/*) echo "resolved to the source tree, not site-packages: $COMPONENTS_DIR" >&2; exit 1 ;; esac && \
    mkdir -p "$COMPONENTS_DIR/Senmayo" && \
    echo "$COMPONENTS_DIR" > /opt/senmayo_components_dir.txt
COPY docker/comp_senmayo_bridge.py /tmp/comp_senmayo_bridge.py
RUN COMPONENTS_DIR=$(cat /opt/senmayo_components_dir.txt) && \
    cp /tmp/comp_senmayo_bridge.py "$COMPONENTS_DIR/Senmayo/comp_senmayo_bridge.py" && \
    touch "$COMPONENTS_DIR/Senmayo/__init__.py"

# --- Trained model artifacts and configs ---
RUN mkdir -p /opt/senmayo/configs
COPY data/processed/bridge_model.joblib /opt/senmayo/bridge_model.joblib
COPY reinvent_run/outputs/run2_stage1.chkpt /opt/senmayo/run2_stage1.chkpt
COPY docker/configs/ /opt/senmayo/configs/
COPY docker/cli.py /opt/senmayo/cli.py

# Sanity check at build time: the bridge model must load and predict.
RUN echo "c1ccccc1" > /tmp/smoke.smi && \
    python /opt/senmayo/cli.py score --input /tmp/smoke.smi --output /tmp/smoke_out.csv && \
    cat /tmp/smoke_out.csv

WORKDIR /data
ENTRYPOINT ["python", "/opt/senmayo/cli.py"]
CMD ["--help"]
