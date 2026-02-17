#!/usr/bin/env python3
"""
Benchmark: Classical ML + Target Features for ASO Datasets (OpenASO & ASOptimizer)

Standalone script version of benchmark_aso_classical.ipynb.
Reproduces paper baselines (Tables A5-A8) and evaluates impact of target gene features.

Usage:
    python experiments/benchmark_aso_classical.py

Outputs (written to experiments/results/):
    - benchmark_report.txt      Readable summary of all results
    - baseline_results.csv      Phase 1: paper replication results
    - target_feature_results.csv Phase 2: all model x variant PCC values
"""

import sys
import logging
import warnings
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file from project root if it exists
_env_file = PROJECT_ROOT / ".env"
if _env_file.is_file():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

from oligogym.data import DatasetDownloader
from oligogym.features import KMersCounts, OneHotEncoder, ModelGeneratorEmbeddings
from oligogym.metrics import regression_metrics
from oligogym.models import (
    LinearModel,
    XGBoostModel,
)
from oligogym.target_features import TargetFeatures, TargetContextEncoder
from oligogym.download_genes import NCBIGeneDownloader

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

# =============================================================================
# Configuration
# =============================================================================
MODELS_TO_RUN = ["Linear", "XGB"]
ASO_DATASET_KEYS = ["OpenASO", "ASOptimizer"]
N_FOLDS = 5
RANDOM_STATE = 42

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
EMBEDDING_CACHE_DIR = str(PROJECT_ROOT / "experiments" / "embeddings_cache")
EMBEDDING_BACKBONE = "aido_rna_1b600m"
EMBEDDING_PCA = True       # Apply PCA to RNA embeddings (set False to use full dims)
EMBEDDING_PCA_DIMS = {     # Per-dataset PCA component options when EMBEDDING_PCA is True
    "OpenASO": [128],
    "ASOptimizer": [256, 512],
}
CONTEXT_WINDOWS = [50, 100]  # Target context window sizes (nt) to sweep

# NCBI Entrez config for auto-downloading missing gene transcripts.
# Set your email here or via the NCBI_EMAIL environment variable.
NCBI_EMAIL = "your.email@example.com"  # <-- CHANGE THIS or set NCBI_EMAIL env var
NCBI_API_KEY = None  # optional, raises NCBI rate limit to 10 req/s

# Featurizer configs
FEAT_CONFIGS = {
    "kmer_1_mod":      {"class": KMersCounts, "kwargs": {"k": [1], "modification_abundance": True}},
    "kmer_12_nomod":   {"class": KMersCounts, "kwargs": {"k": [1, 2], "modification_abundance": False}},
    "kmer_12_mod":     {"class": KMersCounts, "kwargs": {"k": [1, 2], "modification_abundance": True}},
    "kmer_123_mod":    {"class": KMersCounts, "kwargs": {"k": [1, 2, 3], "modification_abundance": True}},
    "kmer_123_nomod":  {"class": KMersCounts, "kwargs": {"k": [1, 2, 3], "modification_abundance": False}},
    "ohe_full":        {"class": OneHotEncoder, "kwargs": {"encode_components": ["base", "sugar", "phosphate"]}},
}

MODEL_CLASSES = {
    "Linear": LinearModel,
    "XGB": XGBoostModel,
}

# Paper Table 2: best PCC per model (random 5-fold CV)
PAPER_RESULTS = {
    "OpenASO": {
        "Linear": (0.33, 0.02), "XGB": (0.27, 0.02),
    },
    "ASOptimizer": {
        "Linear": (0.47, 0.01), "XGB": (0.58, 0.01),
    },
}

# Best configs from paper appendix Tables A5-A8 (Random split column)
BEST_CONFIGS = {
    ("OpenASO",      "Linear"):   ("kmer_12_nomod",  {"task": "regression", "type": "standard"}),
    ("ASOptimizer",  "Linear"):   ("ohe_full",       {"task": "regression", "type": "ridge"}),
    ("OpenASO",      "XGB"):      ("kmer_12_nomod",  {"task": "regression", "n_estimators": 1000, "max_depth": 10}),
    ("ASOptimizer",  "XGB"):      ("kmer_123_mod",   {"task": "regression", "n_estimators": 100,  "max_depth": 10}),
}





# =============================================================================
# Helper Functions
# =============================================================================
def compute_features(X_all, feat_key):
    """Compute features from HELM sequences using the given featurizer config."""
    cfg = FEAT_CONFIGS[feat_key]
    feat = cfg["class"](**cfg["kwargs"])
    if isinstance(feat, OneHotEncoder):
        X_feat = feat.fit_transform(X_all.tolist(), flatten=True)
        return pd.DataFrame(X_feat)
    else:
        return feat.fit_transform(X_all.tolist())


def run_cv(model_class, model_kwargs, X, y, n_folds=N_FOLDS):
    """Run k-fold CV. Returns (mean_pcc, std_pcc)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    pccs = []
    for train_idx, test_idx in kf.split(X):
        if isinstance(X, pd.DataFrame):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        metrics = regression_metrics(y_te, y_pred)
        pccs.append(metrics["pearson_correlation"])
    return np.mean(pccs), np.std(pccs)


def combine_features(*dfs):
    """Concatenate multiple DataFrames/arrays column-wise, resetting indices."""
    parts = []
    for df in dfs:
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        parts.append(df.reset_index(drop=True))
    combined = pd.concat(parts, axis=1)
    combined.columns = [str(i) for i in range(len(combined.columns))]
    return combined


def log(msg, file=None):
    """Print to stdout and optionally write to file."""
    print(msg)
    if file is not None:
        file.write(msg + "\n")


# =============================================================================
# Main
# =============================================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / "benchmark_report.txt"
    baseline_csv_path = RESULTS_DIR / "baseline_results.csv"
    target_csv_path = RESULTS_DIR / "target_feature_results.csv"

    with open(report_path, "w") as report:
        log(f"{'='*70}", report)
        log(f"  ASO Classical ML Benchmark — {timestamp}", report)
        log(f"{'='*70}", report)
        log(f"Models: {MODELS_TO_RUN}", report)
        log(f"Datasets: {ASO_DATASET_KEYS}", report)
        log(f"CV: {N_FOLDS}-fold, random_state={RANDOM_STATE}", report)
        log("", report)

        # -----------------------------------------------------------------
        # Load datasets
        # -----------------------------------------------------------------
        log("Loading datasets...", report)
        downloader = DatasetDownloader()
        datasets = {}
        for key in ASO_DATASET_KEYS:
            data = downloader.download(key)
            datasets[key] = data
            n_targets = len(set(data.targets)) if data.targets is not None else 0
            log(f"  {key}: {len(data.x)} samples, {n_targets} targets, "
                f"y in [{data.y.min():.2f}, {data.y.max():.2f}]", report)
        log("", report)

        # -----------------------------------------------------------------
        # Pre-compute featurizations
        # -----------------------------------------------------------------
        log("Computing baseline features...", report)
        dataset_features = {}
        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            feat_keys_needed = set()
            for model_name in MODELS_TO_RUN:
                feat_key, _ = BEST_CONFIGS[(ds_key, model_name)]
                feat_keys_needed.add(feat_key)
            for feat_key in sorted(feat_keys_needed):
                X_feat = compute_features(data.x, feat_key)
                dataset_features[(ds_key, feat_key)] = X_feat
                log(f"  {ds_key}/{feat_key}: {X_feat.shape}", report)
        log("", report)

        # -----------------------------------------------------------------
        # Phase 1: Replicate paper baselines
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log("  PHASE 1: Replicate Paper Baselines", report)
        log(f"{'='*70}", report)

        baseline_results = {}
        baseline_rows = []

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            y_all = data.y
            log(f"\n  {ds_key} ({len(y_all)} samples)", report)
            log(f"  {'-'*50}", report)

            for model_name in MODELS_TO_RUN:
                feat_key, model_kwargs = BEST_CONFIGS[(ds_key, model_name)]
                X_feat = dataset_features[(ds_key, feat_key)]
                model_class = MODEL_CLASSES[model_name]

                mean_pcc, std_pcc = run_cv(model_class, model_kwargs, X_feat, y_all)
                baseline_results[(ds_key, model_name)] = (mean_pcc, std_pcc)

                paper_mean, paper_std = PAPER_RESULTS[ds_key][model_name]
                delta = mean_pcc - paper_mean
                log(f"    {model_name:8s} [{feat_key:15s}]: "
                    f"PCC = {mean_pcc:.3f} +/- {std_pcc:.3f}  "
                    f"(paper: {paper_mean:.2f}, delta = {delta:+.3f})", report)

                baseline_rows.append({
                    "dataset": ds_key,
                    "model": model_name,
                    "featurizer": feat_key,
                    "pcc_mean": round(mean_pcc, 4),
                    "pcc_std": round(std_pcc, 4),
                    "paper_pcc_mean": paper_mean,
                    "paper_pcc_std": paper_std,
                    "delta_vs_paper": round(delta, 4),
                })

        df_baseline = pd.DataFrame(baseline_rows)
        df_baseline.to_csv(baseline_csv_path, index=False)
        log(f"\n  Saved {len(df_baseline)} baseline results to {baseline_csv_path.name}", report)
        log("", report)

        # -----------------------------------------------------------------
        # Check & download missing target gene transcripts
        # -----------------------------------------------------------------
        ref_dir = str(PROJECT_ROOT / "data" / "reference_transcripts")
        os.makedirs(ref_dir, exist_ok=True)

        # Collect all unique target genes across datasets
        all_target_genes = set()
        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            if data.targets is not None:
                for t in data.targets:
                    if t is not None and not pd.isna(t) and t != "negative_control":
                        all_target_genes.add(t)

        # Find which genes are missing .fna files
        missing_genes = [
            g for g in sorted(all_target_genes)
            if not os.path.isfile(os.path.join(ref_dir, f"{g}.fna"))
        ]

        existing_genes = [
            g for g in sorted(all_target_genes)
            if os.path.isfile(os.path.join(ref_dir, f"{g}.fna"))
        ]

        if missing_genes:
            log(f"{'='*70}", report)
            log(f"  Gene Transcript Check: {len(existing_genes)} found, "
                f"{len(missing_genes)} MISSING in {ref_dir}", report)
            log(f"{'='*70}", report)

            # Resolve email: config constant > env var
            ncbi_email = os.environ.get("NCBI_EMAIL") or NCBI_EMAIL
            ncbi_api_key = os.environ.get("NCBI_API_KEY") or NCBI_API_KEY

            if not ncbi_email or ncbi_email == "your.email@example.com":
                log("", report)
                log("  !!! CANNOT DOWNLOAD — no valid NCBI email configured !!!", report)
                log("  Fix: set NCBI_EMAIL in the Configuration section of this script,", report)
                log("       or export NCBI_EMAIL=you@example.com in your shell.", report)
                log("", report)
                log("  Without gene transcript files, ALL target feature experiments", report)
                log("  will produce identical results (features are constant 0/NaN).", report)
            else:
                log(f"\n  Downloading {len(missing_genes)} gene transcripts "
                    f"via NCBI Entrez...", report)
                log(f"  Email: {ncbi_email}", report)
                log(f"  Output dir: {ref_dir}", report)
                log("", report)

                downloader_ncbi = NCBIGeneDownloader(
                    email=ncbi_email,
                    output_dir=ref_dir,
                    api_key=ncbi_api_key,
                )
                dl_results = downloader_ncbi.process_genes(missing_genes)
                downloaded = [g for g, p in dl_results.items() if p is not None]
                failed = [g for g, p in dl_results.items() if p is None]

                log(f"\n  {'='*50}", report)
                log(f"  Download summary: {len(downloaded)} OK, "
                    f"{len(failed)} failed out of {len(missing_genes)}", report)
                log(f"  {'='*50}", report)
                for g in downloaded:
                    log(f"    [OK]     {g}.fna", report)
                for g in failed:
                    log(f"    [FAILED] {g}.fna", report)
            log("", report)
        else:
            log(f"  All {len(all_target_genes)} target gene transcript files "
                f"present in {ref_dir}", report)

        # -----------------------------------------------------------------
        # Audit: per-dataset impact of missing transcripts
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log("  Missing Transcript Audit", report)
        log(f"{'='*70}", report)

        # Recheck which genes are still missing after download attempt
        still_missing = set(
            g for g in all_target_genes
            if not os.path.isfile(os.path.join(ref_dir, f"{g}.fna"))
        )

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            targets = data.targets
            n_total = len(targets)

            # Per-sample: does the target have a transcript file?
            has_file = [
                t is not None and not pd.isna(t)
                and t != "negative_control"
                and t not in still_missing
                for t in targets
            ]
            n_covered = sum(has_file)
            n_missing = n_total - n_covered

            log(f"\n  {ds_key}: {n_total} samples", report)
            log(f"    Samples with transcript:    {n_covered} "
                f"({n_covered/n_total:.1%})", report)
            log(f"    Samples WITHOUT transcript: {n_missing} "
                f"({n_missing/n_total:.1%})", report)

            # Break down by gene: which missing genes affect how many rows
            if still_missing:
                gene_counts = Counter(
                    t for t in targets
                    if t is not None and not pd.isna(t) and t in still_missing
                )
                # Also count negative controls / NaN
                n_ctrl = sum(
                    1 for t in targets
                    if t is None or (isinstance(t, str) and t == "negative_control")
                    or (not isinstance(t, str) and pd.isna(t))
                )
                if n_ctrl:
                    log(f"    Negative controls / no target: {n_ctrl} samples", report)
                if gene_counts:
                    log(f"    Missing genes breakdown:", report)
                    for gene, cnt in gene_counts.most_common():
                        log(f"      {gene:20s}: {cnt:5d} samples", report)

            # Y-distribution comparison: covered vs missing
            y_all = data.y
            y_covered = y_all[has_file]
            y_missing = y_all[[not h for h in has_file]]
            if len(y_covered) > 0 and len(y_missing) > 0:
                log(f"    Y-distribution (potential bias check):", report)
                log(f"      Covered   — mean: {y_covered.mean():.3f}, "
                    f"std: {y_covered.std():.3f}, "
                    f"median: {np.median(y_covered):.3f}", report)
                log(f"      Missing   — mean: {y_missing.mean():.3f}, "
                    f"std: {y_missing.std():.3f}, "
                    f"median: {np.median(y_missing):.3f}", report)

        log("", report)

        # -----------------------------------------------------------------
        # Compute target features (TF once; CtxKmer per window)
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log(f"  Computing Target Gene Features (windows: {CONTEXT_WINDOWS})", report)
        log(f"{'='*70}", report)

        # dataset_target_feats[ds_key] = {"TF": ..., "coverage": ...,
        #     "CtxKmer": {window: df, ...}}
        dataset_target_feats = {}

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            log(f"\n  {ds_key}: {len(data.x)} samples, "
                f"{len(set(data.targets))} unique targets", report)

            tf = TargetFeatures(reference_dir=ref_dir)
            X_tf = tf.fit_transform(data.x, targets=data.targets)
            coverage = X_tf["has_target_info"].mean()
            log(f"    TF:      {X_tf.shape}, coverage: {coverage:.1%}", report)

            ctx_kmer_by_window = {}
            for w in CONTEXT_WINDOWS:
                tce_kmer = TargetContextEncoder(
                    reference_dir=ref_dir, context_window=w,
                    encoding="kmer", encoding_k=[1, 2, 3], pooling="mean",
                )
                X_ctx_kmer = tce_kmer.fit_transform(data.x, targets=data.targets)
                ctx_kmer_by_window[w] = X_ctx_kmer
                log(f"    CtxKmer (w={w:3d}): {X_ctx_kmer.shape}", report)

            dataset_target_feats[ds_key] = {
                "TF": X_tf.fillna(-1),
                "CtxKmer": ctx_kmer_by_window,
                "coverage": coverage,
            }

        log("", report)

        # -----------------------------------------------------------------
        # Compute RNA embeddings (ASO sequences + target context per window)
        # Then apply PCA for each (dataset, window, pca_dim) combination.
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log(f"  Computing RNA Embeddings ({EMBEDDING_BACKBONE})", report)
        log(f"  Context windows: {CONTEXT_WINDOWS}", report)
        if EMBEDDING_PCA:
            log(f"  PCA dims: {EMBEDDING_PCA_DIMS}", report)
        log(f"{'='*70}", report)

        embedder = ModelGeneratorEmbeddings(
            backbone=EMBEDDING_BACKBONE,
            pooling_strategy="mean",
            batch_size=32,
            cache_dir=EMBEDDING_CACHE_DIR,
        )

        # Storage:
        #   aso_embed_pca[(ds_key, pca_dim)] = DataFrame   (or "raw" key when PCA off)
        #   ctx_embed_pca[(ds_key, window, pca_dim)] = DataFrame
        aso_embed_pca = {}
        ctx_embed_pca = {}

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            pca_dim_list = EMBEDDING_PCA_DIMS[ds_key] if EMBEDDING_PCA else ["raw"]
            log(f"\n  {ds_key}:", report)

            # ASO sequence embeddings (computed once, PCA'd per dim)
            X_embed_raw = embedder.fit_transform(data.x)
            log(f"    ASO embed (raw): {X_embed_raw.shape}", report)

            if EMBEDDING_PCA:
                for pdim in pca_dim_list:
                    n_aso = min(pdim, X_embed_raw.shape[1], X_embed_raw.shape[0])
                    pca_aso = PCA(n_components=n_aso, random_state=RANDOM_STATE)
                    X_aso_pca = pca_aso.fit_transform(X_embed_raw)
                    aso_embed_pca[(ds_key, pdim)] = pd.DataFrame(X_aso_pca)
                    log(f"    ASO PCA (d={pdim}): {X_embed_raw.shape[1]}→{n_aso}, "
                        f"explained var: "
                        f"{pca_aso.explained_variance_ratio_.sum():.1%}", report)
            else:
                aso_embed_pca[(ds_key, "raw")] = pd.DataFrame(X_embed_raw)

            # Context embeddings: compute raw per window, then PCA per dim
            embed_encoder = lambda seqs: embedder.transform(seqs, input_format="fasta")

            for w in CONTEXT_WINDOWS:
                tce_embed = TargetContextEncoder(
                    reference_dir=ref_dir, context_window=w,
                    encoding=embed_encoder, pooling="mean",
                )
                X_ctx_raw = tce_embed.fit_transform(data.x, targets=data.targets)
                log(f"    CtxEmbed (w={w:3d}, raw): {X_ctx_raw.shape}", report)

                if EMBEDDING_PCA:
                    ctx_arr = (X_ctx_raw.values if isinstance(X_ctx_raw, pd.DataFrame)
                               else X_ctx_raw)
                    for pdim in pca_dim_list:
                        n_ctx = min(pdim, ctx_arr.shape[1], ctx_arr.shape[0])
                        pca_ctx = PCA(n_components=n_ctx, random_state=RANDOM_STATE)
                        X_ctx_pca = pca_ctx.fit_transform(ctx_arr)
                        ctx_embed_pca[(ds_key, w, pdim)] = pd.DataFrame(X_ctx_pca)
                        log(f"      PCA (d={pdim}): {ctx_arr.shape[1]}→{n_ctx}, "
                            f"explained var: "
                            f"{pca_ctx.explained_variance_ratio_.sum():.1%}", report)
                else:
                    ctx_embed_pca[(ds_key, w, "raw")] = (
                        X_ctx_raw if isinstance(X_ctx_raw, pd.DataFrame)
                        else pd.DataFrame(X_ctx_raw))

        log("", report)

        # -----------------------------------------------------------------
        # Phase 2 & 3: Run ALL (window, pca_dim) combos
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log("  PHASE 2 & 3: Target Feature + Embedding Experiments", report)
        log(f"  Sweeping context_window × PCA dims", report)
        log(f"{'='*70}", report)

        all_results = {}   # key: (ds_key, model, variant_label) → (mean, std)
        all_rows = []      # flat list for CSV export

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            y_all = data.y
            tf_data = dataset_target_feats[ds_key]
            X_tf = tf_data["TF"]
            pca_dim_list = EMBEDDING_PCA_DIMS[ds_key] if EMBEDDING_PCA else ["raw"]

            log(f"\n{'='*60}", report)
            log(f"  {ds_key} (coverage: {tf_data['coverage']:.1%})", report)
            log(f"{'='*60}", report)

            for model_name in MODELS_TO_RUN:
                feat_key, model_kwargs = BEST_CONFIGS[(ds_key, model_name)]
                X_base = dataset_features[(ds_key, feat_key)]
                model_class = MODEL_CLASSES[model_name]

                # --- Baseline & +TF (invariant to window/pca) ---
                for vname, X_feat in [
                    ("Baseline", X_base),
                    ("+ TF", combine_features(X_base, X_tf)),
                ]:
                    mean_pcc, std_pcc = run_cv(model_class, model_kwargs,
                                               X_feat, y_all)
                    label = vname
                    all_results[(ds_key, model_name, label)] = (mean_pcc, std_pcc)
                    marker = " <-- baseline" if vname == "Baseline" else ""
                    log(f"\n    {model_name} | {label:40s} "
                        f"({X_feat.shape[1]:4d} feats): "
                        f"PCC = {mean_pcc:.3f} +/- {std_pcc:.3f}{marker}", report)

                    paper_mean, paper_std = PAPER_RESULTS[ds_key][model_name]
                    all_rows.append({
                        "dataset": ds_key, "model": model_name,
                        "variant": label, "featurizer": feat_key,
                        "context_window": None, "pca_dim": None,
                        "n_features": X_feat.shape[1],
                        "pcc_mean": round(mean_pcc, 4),
                        "pcc_std": round(std_pcc, 4),
                        "paper_pcc_mean": paper_mean,
                        "paper_pcc_std": paper_std,
                        "delta_vs_paper": round(mean_pcc - paper_mean, 4),
                        "target_coverage": round(tf_data["coverage"], 3),
                    })

                # --- Sweep over (window, pca_dim) combos ---
                for w in CONTEXT_WINDOWS:
                    X_ctx_kmer = tf_data["CtxKmer"][w]

                    # CtxKmer variants (no PCA dependency)
                    for vname, X_feat in [
                        (f"+ CtxKmer (w={w})",
                         combine_features(X_base, X_ctx_kmer)),
                        (f"+ TF + CtxKmer (w={w})",
                         combine_features(X_base, X_tf, X_ctx_kmer)),
                    ]:
                        if vname in [r[1] for r in all_results
                                     if r[0] == ds_key]:
                            continue  # already computed
                        mean_pcc, std_pcc = run_cv(
                            model_class, model_kwargs, X_feat, y_all)
                        all_results[(ds_key, model_name, vname)] = (
                            mean_pcc, std_pcc)
                        log(f"    {model_name} | {vname:40s} "
                            f"({X_feat.shape[1]:4d} feats): "
                            f"PCC = {mean_pcc:.3f} +/- {std_pcc:.3f}", report)

                        paper_mean, paper_std = PAPER_RESULTS[ds_key][model_name]
                        all_rows.append({
                            "dataset": ds_key, "model": model_name,
                            "variant": vname, "featurizer": feat_key,
                            "context_window": w, "pca_dim": None,
                            "n_features": X_feat.shape[1],
                            "pcc_mean": round(mean_pcc, 4),
                            "pcc_std": round(std_pcc, 4),
                            "paper_pcc_mean": paper_mean,
                            "paper_pcc_std": paper_std,
                            "delta_vs_paper": round(mean_pcc - paper_mean, 4),
                            "target_coverage": round(tf_data["coverage"], 3),
                        })

                    for pdim in pca_dim_list:
                        X_ctx_embed = ctx_embed_pca[(ds_key, w, pdim)]
                        X_aso_embed = aso_embed_pca[(ds_key, pdim)]
                        tag = f"w={w},d={pdim}"

                        # Phase 2 style: base featurizer + context embed
                        phase2_variants = [
                            (f"+ CtxEmbed ({tag})",
                             combine_features(X_base, X_ctx_embed)),
                            (f"+ TF + CtxEmbed ({tag})",
                             combine_features(X_base, X_tf, X_ctx_embed)),
                        ]

                        # Phase 3 style: RNA embed as base featurizer
                        phase3_variants = [
                            (f"Embed (d={pdim})",
                             X_aso_embed),
                            (f"Embed (d={pdim}) + TF",
                             combine_features(X_aso_embed, X_tf)),
                            (f"Embed (d={pdim}) + CtxKmer (w={w})",
                             combine_features(X_aso_embed, X_ctx_kmer)),
                            (f"Embed (d={pdim}) + CtxEmbed ({tag})",
                             combine_features(X_aso_embed, X_ctx_embed)),
                        ]

                        for vname, X_feat in phase2_variants + phase3_variants:
                            # Skip if already computed (e.g. Embed(d=X) runs
                            # once per pdim, not per window)
                            rkey = (ds_key, model_name, vname)
                            if rkey in all_results:
                                continue

                            mean_pcc, std_pcc = run_cv(
                                model_class, model_kwargs, X_feat, y_all)
                            all_results[rkey] = (mean_pcc, std_pcc)
                            log(f"    {model_name} | {vname:40s} "
                                f"({X_feat.shape[1]:4d} feats): "
                                f"PCC = {mean_pcc:.3f} +/- {std_pcc:.3f}",
                                report)

                            paper_mean, paper_std = PAPER_RESULTS[ds_key][
                                model_name]
                            featurizer = ("rna_embed" if vname.startswith("Embed")
                                          else feat_key)
                            all_rows.append({
                                "dataset": ds_key, "model": model_name,
                                "variant": vname, "featurizer": featurizer,
                                "context_window": w, "pca_dim": pdim,
                                "n_features": X_feat.shape[1],
                                "pcc_mean": round(mean_pcc, 4),
                                "pcc_std": round(std_pcc, 4),
                                "paper_pcc_mean": paper_mean,
                                "paper_pcc_std": paper_std,
                                "delta_vs_paper": round(
                                    mean_pcc - paper_mean, 4),
                                "target_coverage": round(
                                    tf_data["coverage"], 3),
                            })

        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(target_csv_path, index=False)
        log(f"\n  Saved {len(df_all)} results to {target_csv_path.name}", report)
        log("", report)

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log("  SUMMARY", report)
        log(f"{'='*70}", report)

        for ds_key in ASO_DATASET_KEYS:
            # Gather all variant keys for this dataset
            ds_variants = {v for (d, m, v) in all_results if d == ds_key}

            log(f"\n  {ds_key} (target coverage: "
                f"{dataset_target_feats[ds_key]['coverage']:.1%})", report)

            paper_best_model = max(
                MODELS_TO_RUN, key=lambda m: PAPER_RESULTS[ds_key][m][0])
            paper_best_mean, paper_best_std = PAPER_RESULTS[ds_key][
                paper_best_model]

            for model_name in MODELS_TO_RUN:
                baseline_mean, baseline_std = all_results[
                    (ds_key, model_name, "Baseline")]

                # Find best across all variants for this model
                model_variants = [
                    v for v in ds_variants
                    if (ds_key, model_name, v) in all_results
                ]
                best_var = max(
                    model_variants,
                    key=lambda v: all_results[(ds_key, model_name, v)][0],
                )
                best_mean, best_std = all_results[
                    (ds_key, model_name, best_var)]
                delta = best_mean - baseline_mean

                log(f"\n    {model_name}:", report)
                log(f"      Baseline:   {baseline_mean:.3f} +/- "
                    f"{baseline_std:.3f}  (paper: {PAPER_RESULTS[ds_key][model_name][0]:.2f})",
                    report)
                log(f"      Best:       {best_mean:.3f} +/- {best_std:.3f}  "
                    f"delta={delta:+.3f}  [{best_var}]", report)

            # Overall best across all models
            all_keys = [
                (ds_key, m, v)
                for m in MODELS_TO_RUN for v in ds_variants
                if (ds_key, m, v) in all_results
            ]
            overall_best = max(all_keys, key=lambda k: all_results[k][0])
            ob_mean, ob_std = all_results[overall_best]
            log(f"\n    Overall best: {ob_mean:.3f} +/- {ob_std:.3f}  "
                f"[{overall_best[1]} / {overall_best[2]}]  "
                f"vs paper best ({paper_best_model} "
                f"{paper_best_mean:.2f}): {ob_mean - paper_best_mean:+.3f}",
                report)

        # Top 10 variants across all datasets/models
        log(f"\n  {'='*70}", report)
        log("  Top 10 Configurations (across all datasets × models)", report)
        log(f"  {'='*70}", report)
        sorted_keys = sorted(all_results.keys(),
                             key=lambda k: all_results[k][0], reverse=True)
        for i, key in enumerate(sorted_keys[:10]):
            mean, std = all_results[key]
            log(f"    {i+1:2d}. {mean:.3f} +/- {std:.3f}  "
                f"{key[0]:15s} {key[1]:6s} {key[2]}", report)

        log(f"\n{'='*70}", report)
        log(f"  Done. Results written to {RESULTS_DIR}/", report)
        log(f"    - {report_path.name}", report)
        log(f"    - {baseline_csv_path.name}", report)
        log(f"    - {target_csv_path.name}", report)
        log(f"{'='*70}", report)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
