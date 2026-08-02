"""Fast, batched GPS-based SenMayo reversal oracle.

Methodology: for each compound, predict a continuous per-gene direction
score (P(up) - P(down), median across GPS's 4 cell-line models) across a
~9.7k-gene background (LINCS L1000 BING "best inferred" genes intersected
with GPS's gene-feature coverage). The reversal score is a permutation
Z-score: SenMayo's mean gene-score compared against the mean gene-score of
many random same-size gene sets drawn from the same background, for that
same compound, sign-flipped so that a compound which concentrates SenMayo
genes at the down-regulated end of its profile gets a positive "reversal"
score.

Two earlier approaches were tried and both failed the positive-control
robustness check (Section 3.0: known senolytics should score above the
LINCS library average) -- and diagnosis showed why. (1) A 0.95-confidence
thresholded category average over the 123 SenMayo genes: too coarse, most
genes got "no-changed" for any single compound. (2) An ssGSEA enrichment
score (gseapy) over the ~9.7k-gene background: still failed, because
diagnostic checks showed SenMayo genes get a systematically higher
P(up)-P(down) score than the background for *every* compound tested,
including inert ones (e.g. benzene) -- i.e. GPS's per-gene predictions
carry a compound-independent marginal bias for this particular gene set
(plausibly because SenMayo genes cluster in GO-feature space), which
neither approach corrected for. GPS's own repo (Run_reversal_score.py)
handles exactly this with a permutation background; this module does the
same directly instead of via ssGSEA, which has no equivalent gene-set
permutation null built in.

Public interface: score_compounds(smiles_list) -> np.ndarray of reversal
scores.
"""
import sys

sys.path.insert(0, "/home/alperen/projects/reinvent_geroprotector/gps/GPS/GPS4Drugs/code")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

GPS_ROOT = "/home/alperen/projects/reinvent_geroprotector/gps/GPS/GPS4Drugs/"
CELL_LINES = ["HEPG2_t0", "MCF7_t1", "PC3_t1", "VCAP_t1"]
SENMAYO_PATH = "/home/alperen/projects/reinvent_geroprotector/data/raw/senmayo/senmayo_human_genes.txt"
LINCS_GENE_INFO = "/home/alperen/projects/reinvent_geroprotector/data/raw/lincs/gene_info92.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_ROWS = 400_000  # rows per forward pass (compounds * genes), bounds GPU memory
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 13

_state = {}


def _load():
    if _state:
        return
    gi = pd.read_csv(LINCS_GENE_INFO, sep="\t")
    bing_genes = set(gi[gi["pr_is_bing"] == 1]["pr_gene_symbol"])

    gene_feat_full = pd.read_csv(
        GPS_ROOT + "data/input_gene_features/go_fingerprints_allGenesExt.csv", index_col=0
    )
    bg_genes = sorted(bing_genes & set(gene_feat_full.index))

    senmayo_genes = [l.strip() for l in open(SENMAYO_PATH)]
    senmayo_in_bg = sorted(set(senmayo_genes) & set(bg_genes))
    gene_pos = {g: i for i, g in enumerate(bg_genes)}
    senmayo_idx = np.array([gene_pos[g] for g in senmayo_in_bg])

    n_bg = len(bg_genes)
    rng = np.random.RandomState(PERMUTATION_SEED)
    perm_idx = np.stack(
        [rng.choice(n_bg, size=len(senmayo_idx), replace=False) for _ in range(N_PERMUTATIONS)]
    )  # (N_PERMUTATIONS, senmayo_size)

    models = {}
    for cl in CELL_LINES:
        m = torch.load(GPS_ROOT + f"code/results/{cl}/multi/model.pkl", map_location=DEVICE, weights_only=False)
        model0 = m["model0"].to(DEVICE)
        model0.eval()
        models[cl] = model0

    _state["bg_genes"] = bg_genes
    _state["senmayo_total"] = len(senmayo_genes)
    _state["senmayo_in_bg"] = senmayo_in_bg
    _state["senmayo_idx"] = senmayo_idx
    _state["perm_idx"] = perm_idx
    _state["gene_feat"] = torch.from_numpy(gene_feat_full.loc[bg_genes].values.astype(np.float32)).to(DEVICE)
    _state["models"] = models


def _morgan_fp(smiles, radius=3, n_bits=1024):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
    return np.frombuffer(fp.ToBitString().encode(), dtype="u1").astype(np.float32) - ord("0")


def _predict_gene_scores(smiles_list):
    """Returns (n_valid, n_genes) continuous P(up)-P(down) scores, and the valid mask."""
    gene_feat = _state["gene_feat"]
    models = _state["models"]
    n_genes = gene_feat.shape[0]

    fps = [_morgan_fp(s) for s in smiles_list]
    valid_mask = np.array([fp is not None for fp in fps])
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return np.empty((0, n_genes), dtype=np.float32), valid_mask

    drug_fps = np.stack([fps[i] for i in valid_idx])
    n = drug_fps.shape[0]
    drug_fps_t = torch.from_numpy(drug_fps).to(DEVICE)

    all_scores = np.empty((n, n_genes), dtype=np.float32)
    comp_chunk = max(1, CHUNK_ROWS // n_genes)
    for start in range(0, n, comp_chunk):
        end = min(start + comp_chunk, n)
        nc = end - start
        drug_block = drug_fps_t[start:end].unsqueeze(1).repeat(1, n_genes, 1).reshape(nc * n_genes, -1)
        gene_block = gene_feat.unsqueeze(0).repeat(nc, 1, 1).reshape(nc * n_genes, -1)
        data = torch.cat([drug_block, gene_block], dim=1)

        cl_probs = []
        with torch.no_grad():
            for model0 in models.values():
                logits = model0(data)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                cl_probs.append(probs)
        median_probs = np.median(np.stack(cl_probs), axis=0)  # (nc*n_genes, 3)
        # class order: 0=down, 1=no-change, 2=up (verified against GPS's own catg_assign)
        signed = median_probs[:, 2] - median_probs[:, 0]  # P(up) - P(down)
        all_scores[start:end] = signed.reshape(nc, n_genes)

    return all_scores, valid_mask


def score_compounds(smiles_list, return_valid_mask=False):
    """Score a list of SMILES with a permutation-null SenMayo reversal Z-score.
    Invalid SMILES get NaN."""
    _load()

    gene_scores, valid_mask = _predict_gene_scores(smiles_list)  # (n_valid, n_bg_genes)
    scores = np.full(len(smiles_list), np.nan, dtype=np.float32)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return (scores, valid_mask) if return_valid_mask else scores

    senmayo_idx = _state["senmayo_idx"]
    perm_idx = _state["perm_idx"]  # (N_PERMUTATIONS, senmayo_size)

    senmayo_mean = gene_scores[:, senmayo_idx].mean(axis=1)  # (n_valid,)
    null_means = gene_scores[:, perm_idx].mean(axis=2)  # (n_valid, N_PERMUTATIONS)
    null_mean = null_means.mean(axis=1)
    null_std = null_means.std(axis=1)
    null_std = np.where(null_std < 1e-8, 1e-8, null_std)

    z = (senmayo_mean - null_mean) / null_std
    reversal = -z  # sign-flip: SenMayo enriched "up" relative to random gene sets -> bad -> negative score

    scores[valid_idx] = reversal
    return (scores, valid_mask) if return_valid_mask else scores


def score_compound(smiles):
    """Single-SMILES convenience wrapper matching AGENTS.md's abstract interface."""
    return float(score_compounds([smiles])[0])


if __name__ == "__main__":
    import time

    t0 = time.time()
    test = {
        "quercetin": "Oc1cc(O)c2c(c1)oc(-c1ccc(O)c(O)c1)c(O)c2=O",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "invalid": "not_a_smiles",
    }
    scores = score_compounds(list(test.values()))
    for name, sc in zip(test.keys(), scores):
        print(f"{name}: {sc}")
    print(f"background genes: {len(_state['bg_genes'])}, SenMayo in background: {len(_state['senmayo_in_bg'])}/{_state['senmayo_total']}")
    print(f"elapsed: {time.time() - t0:.2f}s")
