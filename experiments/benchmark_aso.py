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
from sklearn.model_selection import KFold

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from oligogym.data import DatasetDownloader
from oligogym.features import KMersCounts, OneHotEncoder, ModelGeneratorEmbeddings
from oligogym.metrics import regression_metrics
from oligogym.models import (
    LinearModel,
    NearestNeighborsModel,
    RandomForestModel,
    XGBoostModel,
)
from oligogym.target_features import TargetFeatures, TargetContextEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

# =============================================================================
# Configuration
# =============================================================================
MODELS_TO_RUN = ["Linear", "KNN", "RF", "XGB"]
ASO_DATASET_KEYS = ["OpenASO", "ASOptimizer"]
N_FOLDS = 5
RANDOM_STATE = 42

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
EMBEDDING_CACHE_DIR = str(PROJECT_ROOT / "experiments" / "embeddings_cache")
EMBEDDING_BACKBONE = "aido_rna_650m"

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
    "KNN": NearestNeighborsModel,
    "RF": RandomForestModel,
    "XGB": XGBoostModel,
}

# Paper Table 2: best PCC per model (random 5-fold CV)
PAPER_RESULTS = {
    "OpenASO": {
        "Linear": (0.33, 0.02), "KNN": (0.32, 0.01),
        "RF": (0.35, 0.04), "XGB": (0.27, 0.02),
    },
    "ASOptimizer": {
        "Linear": (0.47, 0.01), "KNN": (0.63, 0.01),
        "RF": (0.64, 0.01), "XGB": (0.58, 0.01),
    },
}

# Best configs from paper appendix Tables A5-A8 (Random split column)
BEST_CONFIGS = {
    ("OpenASO",      "Linear"): ("kmer_12_nomod",  {"task": "regression", "type": "standard"}),
    ("ASOptimizer",  "Linear"): ("ohe_full",       {"task": "regression", "type": "ridge"}),
    ("OpenASO",      "KNN"): ("kmer_123_mod", {"task": "regression", "n_neighbors": 10}),
    ("ASOptimizer",  "KNN"): ("kmer_123_mod", {"task": "regression", "n_neighbors": 10}),
    ("OpenASO",      "RF"): ("kmer_123_mod", {"task": "regression", "n_estimators": 500,  "max_depth": 10}),
    ("ASOptimizer",  "RF"): ("kmer_123_mod", {"task": "regression", "n_estimators": 500,  "max_depth": 20}),
    ("OpenASO",      "XGB"): ("kmer_12_nomod", {"task": "regression", "n_estimators": 1000, "max_depth": 10}),
    ("ASOptimizer",  "XGB"): ("kmer_123_mod",  {"task": "regression", "n_estimators": 100,  "max_depth": 10}),
}

VARIANT_NAMES = [
    "Baseline", "+ TF", "+ CtxKmer", "+ CtxOH", "+ CtxEmbed",
    "+ TF + CtxKmer", "+ TF + CtxOH", "+ TF + CtxEmbed",
]

EMBED_VARIANT_NAMES = [
    "Embed", "Embed + TF", "Embed + CtxKmer", "Embed + CtxOH", "Embed + CtxEmbed",
    "Embed + TF + CtxEmbed",
]

ALL_VARIANT_NAMES = VARIANT_NAMES + EMBED_VARIANT_NAMES


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
        # Compute target features
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log("  Computing Target Gene Features", report)
        log(f"{'='*70}", report)

        ref_dir = str(PROJECT_ROOT / "data" / "reference_transcripts")
        dataset_target_feats = {}

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            log(f"\n  {ds_key}: {len(data.x)} samples, "
                f"{len(set(data.targets))} unique targets", report)

            tf = TargetFeatures(reference_dir=ref_dir)
            X_tf = tf.fit_transform(data.x, targets=data.targets)
            coverage = X_tf["has_target_info"].mean()
            log(f"    TF:      {X_tf.shape}, coverage: {coverage:.1%}", report)

            tce_kmer = TargetContextEncoder(
                reference_dir=ref_dir, context_window=50,
                encoding="kmer", encoding_k=[1, 2, 3], pooling="mean",
            )
            X_ctx_kmer = tce_kmer.fit_transform(data.x, targets=data.targets)
            log(f"    CtxKmer: {X_ctx_kmer.shape}", report)

            tce_oh = TargetContextEncoder(
                reference_dir=ref_dir, context_window=50,
                encoding="onehot", pooling="mean",
            )
            X_ctx_oh = tce_oh.fit_transform(data.x, targets=data.targets)
            log(f"    CtxOH:   {X_ctx_oh.shape}", report)

            dataset_target_feats[ds_key] = {
                "TF": X_tf.fillna(-1),
                "CtxKmer": X_ctx_kmer,
                "CtxOH": X_ctx_oh,
                "coverage": coverage,
            }

        log("", report)

        # -----------------------------------------------------------------
        # Compute RNA embeddings (ASO sequences + target context)
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log(f"  Computing RNA Embeddings ({EMBEDDING_BACKBONE})", report)
        log(f"{'='*70}", report)

        embedder = ModelGeneratorEmbeddings(
            backbone=EMBEDDING_BACKBONE,
            pooling_strategy="mean",
            batch_size=32,
            cache_dir=EMBEDDING_CACHE_DIR,
        )

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            log(f"\n  {ds_key}:", report)

            # ASO sequence embeddings
            X_embed = embedder.fit_transform(data.x)
            dataset_features[(ds_key, "rna_embed")] = pd.DataFrame(X_embed)
            log(f"    ASO embed:  {X_embed.shape}", report)

            # Target context RNA embeddings
            embed_encoder = lambda seqs: embedder.transform(seqs, input_format="fasta")
            tce_embed = TargetContextEncoder(
                reference_dir=ref_dir, context_window=50,
                encoding=embed_encoder, pooling="mean",
            )
            X_ctx_embed = tce_embed.fit_transform(data.x, targets=data.targets)
            dataset_target_feats[ds_key]["CtxEmbed"] = X_ctx_embed
            log(f"    CtxEmbed:   {X_ctx_embed.shape}", report)

        log("", report)

        # -----------------------------------------------------------------
        # Phase 2: Target feature experiments
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log("  PHASE 2: Target Feature Experiments", report)
        log(f"{'='*70}", report)

        all_results = {}
        target_rows = []

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            y_all = data.y
            tf_data = dataset_target_feats[ds_key]
            X_tf = tf_data["TF"]
            X_ctx_kmer = tf_data["CtxKmer"]
            X_ctx_oh = tf_data["CtxOH"]
            X_ctx_embed = tf_data["CtxEmbed"]

            log(f"\n  {ds_key} (coverage: {tf_data['coverage']:.1%})", report)
            log(f"  {'-'*50}", report)

            for model_name in MODELS_TO_RUN:
                feat_key, model_kwargs = BEST_CONFIGS[(ds_key, model_name)]
                X_base = dataset_features[(ds_key, feat_key)]
                model_class = MODEL_CLASSES[model_name]

                variants = {
                    "Baseline":       X_base,
                    "+ TF":           combine_features(X_base, X_tf),
                    "+ CtxKmer":      combine_features(X_base, X_ctx_kmer),
                    "+ CtxOH":        combine_features(X_base, X_ctx_oh),
                    "+ CtxEmbed":     combine_features(X_base, X_ctx_embed),
                    "+ TF + CtxKmer": combine_features(X_base, X_tf, X_ctx_kmer),
                    "+ TF + CtxOH":   combine_features(X_base, X_tf, X_ctx_oh),
                    "+ TF + CtxEmbed": combine_features(X_base, X_tf, X_ctx_embed),
                }

                log(f"\n    {model_name} ({feat_key}):", report)
                for variant_name, X_feat in variants.items():
                    mean_pcc, std_pcc = run_cv(model_class, model_kwargs, X_feat, y_all)
                    all_results[(ds_key, model_name, variant_name)] = (mean_pcc, std_pcc)
                    marker = " <-- baseline" if variant_name == "Baseline" else ""
                    log(f"      {variant_name:18s} ({X_feat.shape[1]:4d} feats): "
                        f"PCC = {mean_pcc:.3f} +/- {std_pcc:.3f}{marker}", report)

                    paper_mean, paper_std = PAPER_RESULTS[ds_key][model_name]
                    target_rows.append({
                        "dataset": ds_key,
                        "model": model_name,
                        "variant": variant_name,
                        "featurizer": feat_key,
                        "n_features": X_feat.shape[1],
                        "pcc_mean": round(mean_pcc, 4),
                        "pcc_std": round(std_pcc, 4),
                        "paper_pcc_mean": paper_mean,
                        "paper_pcc_std": paper_std,
                        "delta_vs_paper": round(mean_pcc - paper_mean, 4),
                        "target_coverage": round(tf_data["coverage"], 3),
                    })

        df_target = pd.DataFrame(target_rows)
        df_target.to_csv(target_csv_path, index=False)
        log(f"\n  Saved {len(df_target)} results to {target_csv_path.name}", report)
        log("", report)

        # -----------------------------------------------------------------
        # Phase 3: RNA Embeddings as ASO Featurizer
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log(f"  PHASE 3: RNA Embedding as ASO Featurizer ({EMBEDDING_BACKBONE})", report)
        log(f"{'='*70}", report)

        embed_rows = []

        for ds_key in ASO_DATASET_KEYS:
            data = datasets[ds_key]
            y_all = data.y
            tf_data = dataset_target_feats[ds_key]
            X_tf = tf_data["TF"]
            X_ctx_kmer = tf_data["CtxKmer"]
            X_ctx_oh = tf_data["CtxOH"]
            X_ctx_embed = tf_data["CtxEmbed"]
            X_embed = dataset_features[(ds_key, "rna_embed")]

            log(f"\n  {ds_key} (coverage: {tf_data['coverage']:.1%})", report)
            log(f"  {'-'*50}", report)

            for model_name in MODELS_TO_RUN:
                _, model_kwargs = BEST_CONFIGS[(ds_key, model_name)]
                model_class = MODEL_CLASSES[model_name]

                embed_variants = {
                    "Embed":              X_embed,
                    "Embed + TF":         combine_features(X_embed, X_tf),
                    "Embed + CtxKmer":    combine_features(X_embed, X_ctx_kmer),
                    "Embed + CtxOH":      combine_features(X_embed, X_ctx_oh),
                    "Embed + CtxEmbed":   combine_features(X_embed, X_ctx_embed),
                    "Embed + TF + CtxEmbed": combine_features(X_embed, X_tf, X_ctx_embed),
                }

                log(f"\n    {model_name} (rna_embed):", report)
                for variant_name, X_feat in embed_variants.items():
                    mean_pcc, std_pcc = run_cv(model_class, model_kwargs, X_feat, y_all)
                    all_results[(ds_key, model_name, variant_name)] = (mean_pcc, std_pcc)
                    log(f"      {variant_name:24s} ({X_feat.shape[1]:4d} feats): "
                        f"PCC = {mean_pcc:.3f} +/- {std_pcc:.3f}", report)

                    paper_mean, paper_std = PAPER_RESULTS[ds_key][model_name]
                    embed_rows.append({
                        "dataset": ds_key,
                        "model": model_name,
                        "variant": variant_name,
                        "featurizer": "rna_embed",
                        "n_features": X_feat.shape[1],
                        "pcc_mean": round(mean_pcc, 4),
                        "pcc_std": round(std_pcc, 4),
                        "paper_pcc_mean": paper_mean,
                        "paper_pcc_std": paper_std,
                        "delta_vs_paper": round(mean_pcc - paper_mean, 4),
                        "target_coverage": round(tf_data["coverage"], 3),
                    })

        # Append Phase 3 results to the target features CSV
        df_embed = pd.DataFrame(embed_rows)
        df_all_results = pd.concat([df_target, df_embed], ignore_index=True)
        df_all_results.to_csv(target_csv_path, index=False)
        log(f"\n  Saved {len(df_all_results)} total results to {target_csv_path.name}", report)
        log("", report)

        # -----------------------------------------------------------------
        # Summary tables
        # -----------------------------------------------------------------
        log(f"{'='*70}", report)
        log("  SUMMARY", report)
        log(f"{'='*70}", report)

        # Per-dataset summary (Phase 2: base featurizer variants)
        for ds_key in ASO_DATASET_KEYS:
            log(f"\n  {ds_key} (target coverage: "
                f"{dataset_target_feats[ds_key]['coverage']:.1%})", report)
            log(f"  {'Model':8s} | {'Baseline':14s} | {'Best (base feat)':24s} | "
                f"{'Best PCC':14s} | {'Delta':8s}", report)
            log(f"  {'-'*80}", report)

            for model_name in MODELS_TO_RUN:
                baseline_mean = all_results[(ds_key, model_name, "Baseline")][0]
                baseline_std = all_results[(ds_key, model_name, "Baseline")][1]
                best_var = max(
                    VARIANT_NAMES[1:],
                    key=lambda v: all_results[(ds_key, model_name, v)][0],
                )
                best_mean, best_std = all_results[(ds_key, model_name, best_var)]
                delta = best_mean - baseline_mean
                log(f"  {model_name:8s} | {baseline_mean:.3f} +/- {baseline_std:.3f} | "
                    f"{best_var:24s} | {best_mean:.3f} +/- {best_std:.3f} | "
                    f"{delta:+.3f}", report)

        # Per-dataset summary (Phase 3: RNA embed variants)
        log(f"\n  {'='*70}", report)
        log("  Phase 3 Summary: RNA Embedding as ASO Featurizer", report)
        log(f"  {'='*70}", report)

        for ds_key in ASO_DATASET_KEYS:
            log(f"\n  {ds_key}", report)
            log(f"  {'Model':8s} | {'Embed Only':14s} | {'Best Embed Variant':24s} | "
                f"{'Best PCC':14s} | {'vs Baseline':10s}", report)
            log(f"  {'-'*80}", report)

            for model_name in MODELS_TO_RUN:
                baseline_mean = all_results[(ds_key, model_name, "Baseline")][0]
                embed_mean = all_results[(ds_key, model_name, "Embed")][0]
                embed_std = all_results[(ds_key, model_name, "Embed")][1]
                best_var = max(
                    EMBED_VARIANT_NAMES,
                    key=lambda v: all_results[(ds_key, model_name, v)][0],
                )
                best_mean, best_std = all_results[(ds_key, model_name, best_var)]
                delta = best_mean - baseline_mean
                log(f"  {model_name:8s} | {embed_mean:.3f} +/- {embed_std:.3f} | "
                    f"{best_var:24s} | {best_mean:.3f} +/- {best_std:.3f} | "
                    f"{delta:+.3f}", report)

        # Overall best comparison (across ALL variants)
        log(f"\n  {'='*70}", report)
        log("  Paper vs Our Best (all variants including RNA embeddings)", report)
        log(f"  {'='*70}", report)
        log(f"  {'Dataset':15s} | {'Paper Best':22s} | {'Our Baseline':22s} | "
            f"{'Our Overall Best':32s} | {'Impr':6s}", report)
        log(f"  {'-'*105}", report)

        for ds_key in ASO_DATASET_KEYS:
            paper_best_model = max(MODELS_TO_RUN, key=lambda m: PAPER_RESULTS[ds_key][m][0])
            paper_best_mean, paper_best_std = PAPER_RESULTS[ds_key][paper_best_model]

            bl_best_model = max(MODELS_TO_RUN,
                                key=lambda m: all_results[(ds_key, m, "Baseline")][0])
            bl_best_mean, bl_best_std = all_results[(ds_key, bl_best_model, "Baseline")]

            best_key = max(
                [(ds_key, m, v) for m in MODELS_TO_RUN for v in ALL_VARIANT_NAMES],
                key=lambda k: all_results[k][0],
            )
            best_mean, best_std = all_results[best_key]
            best_model = best_key[1]
            best_variant = best_key[2]

            log(f"  {ds_key:15s} | {paper_best_mean:.2f} +/- {paper_best_std:.2f} "
                f"({paper_best_model:6s}) | {bl_best_mean:.3f} +/- {bl_best_std:.3f} "
                f"({bl_best_model:6s}) | {best_mean:.3f} +/- {best_std:.3f} "
                f"({best_model} {best_variant}) | {best_mean - paper_best_mean:+.3f}", report)

        # Variant win counts (across ALL variants)
        log(f"\n  Best variant wins (across all dataset x model combos):", report)
        variant_wins = Counter()
        for ds_key in ASO_DATASET_KEYS:
            for model_name in MODELS_TO_RUN:
                best_var = max(
                    ALL_VARIANT_NAMES,
                    key=lambda v: all_results[(ds_key, model_name, v)][0],
                )
                variant_wins[best_var] += 1

        total = len(ASO_DATASET_KEYS) * len(MODELS_TO_RUN)
        log(f"  Total combos: {total}", report)
        for var, count in variant_wins.most_common():
            log(f"    {var:18s}: {count} wins", report)

        log(f"\n{'='*70}", report)
        log(f"  Done. Results written to {RESULTS_DIR}/", report)
        log(f"    - {report_path.name}", report)
        log(f"    - {baseline_csv_path.name}", report)
        log(f"    - {target_csv_path.name}", report)
        log(f"{'='*70}", report)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
