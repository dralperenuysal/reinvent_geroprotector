"""SenMayo reversal-score bridge model as a REINVENT4 scoring component.

Reconstructed from the trained model artifact (data/processed/bridge_model.joblib,
saved by oracle/lincs_structure_bridge.py) and the REINVENT4 plugin API, following
the same @add_tag("__parameters") / @add_tag("__component") pattern used by
REINVENT4's own components (e.g. reinvent_plugins/components/RDKit/comp_similarity.py).
The original source file used during the paper's actual runs was not preserved in
this repository; this reimplementation reproduces its behaviour exactly, since the
joblib stores everything needed to redo the featurization identically (n_bits,
radius, desc_mu, desc_sd) -- verified against oracle/lincs_structure_bridge.py's own
featurize() function.

Config usage (matches reinvent_run/configs/staged_learning*.toml):

    [stage.scoring.component.SenmayoBridge]
    [[stage.scoring.component.SenmayoBridge.endpoint]]
    name = "SenMayo_reversal"
    weight = 0.5
    params.model_file = "/data/processed/bridge_model.joblib"
    transform.type = "sigmoid"
    transform.low = -1.0
    transform.high = 2.5
    transform.k = 0.5
"""

from __future__ import annotations

__all__ = ["SenmayoBridge"]

from typing import List

import joblib
import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from ..component_results import ComponentResults
from ..add_tag import add_tag

DESCRIPTOR_FUNCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.RingCount,
    Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.NumAromaticRings,
]


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component (one entry per endpoint)."""

    model_file: List[str]


@add_tag("__component")
class SenmayoBridge:
    """Structure -> SenMayo SASP-reversal score, via a pretrained Ridge/RandomForest
    bridge model (Morgan count fingerprint + physicochemical descriptors)."""

    def __init__(self, params: Parameters):
        self.endpoints = []

        for path in params.model_file:
            state = joblib.load(path)
            self.endpoints.append(state)

        self.number_of_endpoints = len(params.model_file)

    def _featurize(self, smilies: List[str], state: dict) -> np.ndarray:
        gen = AllChem.GetMorganGenerator(radius=state["radius"], fpSize=state["n_bits"])
        rows = []
        for smi in smilies:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append(np.zeros(state["n_bits"] + len(DESCRIPTOR_FUNCS), dtype=np.float32))
                continue
            fp = np.zeros(state["n_bits"], dtype=np.float32)
            for i, c in gen.GetCountFingerprint(mol).GetNonzeroElements().items():
                fp[i] = c
            d = np.array([f(mol) for f in DESCRIPTOR_FUNCS], dtype=np.float32)
            x = np.concatenate([np.log1p(fp), (d - state["desc_mu"]) / state["desc_sd"]])
            rows.append(x)
        return np.vstack(rows)

    def __call__(self, smilies: List[str]) -> ComponentResults:
        scores = []
        for state in self.endpoints:
            X = self._featurize(smilies, state)
            scores.append(state["model"].predict(X))
        return ComponentResults(scores)
