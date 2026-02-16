"""
Target-Derived Features
=======================
A featurizer that enriches ASO representations with information about WHERE
on the target mRNA transcript each ASO binds and WHAT the local sequence
environment looks like around the binding site.

For each ASO, the featurizer:
1. Looks up the target gene's reference transcripts (.fna file)
2. Reverse complements the ASO base sequence
3. Searches for the binding site on the canonical transcript
4. Computes ~53 features from the match position and local context

Features are returned as a pd.DataFrame that can be concatenated (hstacked)
with any other feature matrix (k-mers, one-hot, embeddings, etc.).

For samples where target info is unavailable (missing .fna, no match,
negative controls), `has_target_info` is set to 0 and all other target
features are NaN. The consumer is responsible for filling NaN values
(e.g., -1 for tree models, column median for linear models).
"""

import os
import warnings
from collections import defaultdict
from itertools import product as itertools_product
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .features import _extract_monomers
from .transcript_matching import (
    DINUCLEOTIDES,
    TranscriptInfo,
    build_transcript_info,
    dinucleotide_frequencies,
    find_exact_matches,
    gc_content,
    match_aso_to_transcripts,
    max_homopolymer_length,
    reverse_complement,
    shannon_entropy,
)


_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


def _resolve_reference_dir(reference_dir: str) -> str:
    """Resolve reference_dir to an absolute path.

    If the path is relative and doesn't exist from the cwd, try resolving
    it relative to the repository root so it works from any working directory
    (e.g., from notebooks/).
    """
    if os.path.isabs(reference_dir):
        return reference_dir
    if os.path.isdir(reference_dir):
        return reference_dir
    repo_relative = str(_REPO_ROOT / reference_dir)
    if os.path.isdir(repo_relative):
        return repo_relative
    return reference_dir


def _helm_to_fasta(oligo_helm: str) -> str:
    """Extract the DNA base sequence from a HELM notation string.

    Uses RNA1 strand only (the antisense strand for single-strand ASOs).
    Modified bases are reduced to their canonical base letter.
    """
    try:
        monomers = _extract_monomers(oligo_helm, strands=["RNA1"])
    except (AssertionError, Exception):
        # Fall back to extracting without strand filter
        monomers = _extract_monomers(oligo_helm)

    monomers["base"] = monomers["base"].replace("EMPTY", "")
    monomers["base"] = monomers["base"].str[-1]
    fasta_str = monomers["base"].str.cat()
    return fasta_str


def _build_nan_row() -> dict:
    """Return a feature dict with has_target_info=0 and all others NaN."""
    row = {
        "has_target_info": 0.0,
        "relative_position": np.nan,
        "binding_region_encoded": np.nan,
        "dist_to_cds_start": np.nan,
        "dist_to_cds_end": np.nan,
        "dist_to_transcript_end": np.nan,
        "num_isoforms_matched": np.nan,
        "targets_spliced_region": np.nan,
        "transcript_length": np.nan,
        "transcript_gc_content": np.nan,
        "utr5_length": np.nan,
        "utr3_length": np.nan,
        "cds_length": np.nan,
        "target_site_gc": np.nan,
        "upstream_flank_gc": np.nan,
        "downstream_flank_gc": np.nan,
        "gc_contrast_upstream": np.nan,
        "gc_contrast_downstream": np.nan,
        "local_region_gc": np.nan,
        "local_region_entropy": np.nan,
        "homopolymer_max_len": np.nan,
    }
    for d in DINUCLEOTIDES:
        row[f"upstream_dinuc_{d}"] = np.nan
        row[f"downstream_dinuc_{d}"] = np.nan
    return row


class TargetFeatures:
    """Featurizer that computes target-transcript-derived features for ASOs.

    For each ASO, looks up the target gene's reference transcript FASTA file,
    finds where the ASO binds, and computes ~53 features describing the
    binding position, local sequence context, and transcript annotations.

    Args:
        reference_dir: Path to directory containing {GENE}.fna files.
        flank_size: Number of nucleotides upstream/downstream of the binding
            site to use for local sequence context features.

    Example::

        from oligogym.data import DatasetDownloader
        from oligogym.features import KMersCounts, TargetFeatures

        data = downloader.download("ASOptimizer")
        X_train, X_test, y_train, y_test, train_idx, test_idx = data.split(
            split_strategy="random", return_index=True
        )

        kmer_feat = KMersCounts(k=[1, 2], modification_abundance=True)
        target_feat = TargetFeatures(reference_dir="data/reference_transcripts")

        X_kmer_train = kmer_feat.fit_transform(X_train.tolist())
        X_target_train = target_feat.fit_transform(X_train, targets=data.targets[train_idx])

        X_combined_train = pd.concat(
            [X_kmer_train.reset_index(drop=True), X_target_train.reset_index(drop=True)],
            axis=1,
        ).fillna(-1)
    """

    def __init__(
        self,
        reference_dir: str = "data/reference_transcripts",
        flank_size: int = 30,
    ):
        self.reference_dir = _resolve_reference_dir(reference_dir)
        self.flank_size = flank_size
        self._gene_cache: dict[str, Optional[TranscriptInfo]] = {}
        self._columns: list[str] = []

    def _load_gene(self, gene_name: str) -> Optional[TranscriptInfo]:
        """Load and cache transcript info for a gene."""
        if gene_name not in self._gene_cache:
            fasta_path = os.path.join(
                self.reference_dir, f"{gene_name}.fna"
            )
            self._gene_cache[gene_name] = build_transcript_info(fasta_path)
        return self._gene_cache[gene_name]

    def _compute_features(
        self,
        aso_fasta: str,
        target_gene: str,
    ) -> dict:
        """Compute all target-derived features for one ASO-target pair."""
        # Handle special targets
        if (
            target_gene is None
            or target_gene == "negative_control"
            or pd.isna(target_gene)
        ):
            return _build_nan_row()

        info = self._load_gene(target_gene)

        # No .fna file
        if info is None:
            return _build_nan_row()

        # Match ASO to transcripts
        match = match_aso_to_transcripts(aso_fasta, info)

        if not match.matched:
            return _build_nan_row()

        # --- Build feature dict ---
        features = {}
        pos = match.match_pos
        aso_len = len(aso_fasta)
        seq = info.sequence

        features["has_target_info"] = 1.0

        # Positional features
        features["relative_position"] = pos / info.length
        features["dist_to_transcript_end"] = info.length - (pos + aso_len)

        # CDS-dependent features
        if info.cds_start is not None and info.cds_end is not None:
            binding_center = pos + aso_len // 2
            if binding_center < info.cds_start:
                features["binding_region_encoded"] = 0.0  # 5'UTR
            elif binding_center < info.cds_end:
                features["binding_region_encoded"] = 1.0  # CDS
            else:
                features["binding_region_encoded"] = 2.0  # 3'UTR

            features["dist_to_cds_start"] = float(pos - info.cds_start)
            features["dist_to_cds_end"] = float((pos + aso_len) - info.cds_end)
        else:
            features["binding_region_encoded"] = np.nan
            features["dist_to_cds_start"] = np.nan
            features["dist_to_cds_end"] = np.nan

        # Isoform features
        features["num_isoforms_matched"] = float(match.num_isoforms_matched)
        features["targets_spliced_region"] = (
            1.0 if match.num_isoforms_matched < match.total_isoforms else 0.0
        )

        # Transcript-level features
        features["transcript_length"] = float(info.length)
        features["transcript_gc_content"] = info.gc_content
        features["utr5_length"] = (
            float(info.utr5_length) if info.utr5_length is not None else np.nan
        )
        features["utr3_length"] = (
            float(info.utr3_length) if info.utr3_length is not None else np.nan
        )
        features["cds_length"] = (
            float(info.cds_length) if info.cds_length is not None else np.nan
        )

        # Local sequence context
        target_site = seq[pos : pos + aso_len]
        features["target_site_gc"] = gc_content(target_site)

        upstream_start = max(0, pos - self.flank_size)
        upstream_flank = seq[upstream_start:pos]
        downstream_end = min(info.length, pos + aso_len + self.flank_size)
        downstream_flank = seq[pos + aso_len : downstream_end]

        features["upstream_flank_gc"] = gc_content(upstream_flank)
        features["downstream_flank_gc"] = gc_content(downstream_flank)
        features["gc_contrast_upstream"] = (
            features["target_site_gc"] - features["upstream_flank_gc"]
        )
        features["gc_contrast_downstream"] = (
            features["target_site_gc"] - features["downstream_flank_gc"]
        )

        # Dinucleotide frequencies
        up_dinucs = dinucleotide_frequencies(upstream_flank)
        for d, freq in up_dinucs.items():
            features[f"upstream_dinuc_{d}"] = freq

        down_dinucs = dinucleotide_frequencies(downstream_flank)
        for d, freq in down_dinucs.items():
            features[f"downstream_dinuc_{d}"] = freq

        # Local region features (upstream + target + downstream)
        local_start = max(0, pos - self.flank_size)
        local_end = min(info.length, pos + aso_len + self.flank_size)
        local_region = seq[local_start:local_end]

        features["local_region_gc"] = gc_content(local_region)
        features["local_region_entropy"] = shannon_entropy(local_region)
        features["homopolymer_max_len"] = float(max_homopolymer_length(local_region))

        return features

    def fit_transform(
        self,
        oligo_list: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        """Parse reference transcripts, build per-gene cache, and compute features.

        Args:
            oligo_list: HELM strings (list or numpy array).
            targets: Target gene names, parallel to oligo_list.

        Returns:
            pd.DataFrame with one row per oligo and ~53 feature columns.
        """
        if isinstance(oligo_list, np.ndarray):
            oligo_list = oligo_list.tolist()
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()

        assert len(oligo_list) == len(targets), (
            f"oligo_list ({len(oligo_list)}) and targets ({len(targets)}) "
            "must have same length"
        )

        # Reset and pre-populate gene cache
        self._gene_cache = {}
        unique_genes = set(
            t
            for t in targets
            if t is not None and not pd.isna(t) and t != "negative_control"
        )
        genes_found = []
        genes_missing = []
        for gene in sorted(unique_genes):
            info = self._load_gene(gene)
            if info is not None:
                genes_found.append(
                    f"  [found]   {gene} -> {info.accession} ({info.length} bp, "
                    f"CDS {info.cds_start}-{info.cds_end if info.cds_end else '?'})"
                )
            else:
                genes_missing.append(f"  [missing]  {gene} -> no .fna file")

        print(f"TargetFeatures: {len(unique_genes)} unique gene(s), "
              f"{len(genes_found)} with .fna, {len(genes_missing)} without")
        for g in genes_found:
            print(g)
        for g in genes_missing:
            print(g)

        if not genes_found:
            warnings.warn(
                "TargetFeatures: no reference transcript files found in "
                f"'{self.reference_dir}'. All target features will be NaN."
            )

        # Compute features for all samples
        rows = []
        for helm, target in zip(oligo_list, targets):
            aso_fasta = _helm_to_fasta(helm)
            row = self._compute_features(aso_fasta, target)
            rows.append(row)

        df = pd.DataFrame(rows)
        self._columns = df.columns.tolist()
        return df

    def transform(
        self,
        oligo_list: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        """Compute features using the gene cache built during fit_transform.

        New genes not seen during fit_transform are loaded on-the-fly.

        Args:
            oligo_list: HELM strings.
            targets: Target gene names, parallel to oligo_list.

        Returns:
            pd.DataFrame with same columns as fit_transform output.
        """
        if isinstance(oligo_list, np.ndarray):
            oligo_list = oligo_list.tolist()
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()

        assert len(oligo_list) == len(targets), (
            f"oligo_list ({len(oligo_list)}) and targets ({len(targets)}) "
            "must have same length"
        )

        rows = []
        for helm, target in zip(oligo_list, targets):
            aso_fasta = _helm_to_fasta(helm)
            row = self._compute_features(aso_fasta, target)
            rows.append(row)

        df = pd.DataFrame(rows)
        if self._columns:
            df = df.reindex(columns=self._columns)
        return df


class TargetContextEncoder:
    """Featurizer that encodes local mRNA context around ASO binding sites.

    For each ASO, finds all exact binding sites across all transcript isoforms
    of the target gene, extracts a context window around each site, encodes
    the context sequences, and pools across binding sites to produce a
    fixed-size feature vector.

    This captures the actual target sequence environment (not just scalar
    summaries) and handles isoform multiplicity via pooling.

    Args:
        reference_dir: Path to directory containing {GENE}.fna files.
        context_window: Number of nucleotides upstream and downstream of the
            binding site to include in the context.
        pooling: Aggregation strategy over multiple binding sites.
            One of ``"mean"``, ``"max"``, or ``"mean_max"`` (concatenation).
        encoding: How to encode each context sequence. One of:

            - ``"kmer"``: k-mer counting (configure k via *encoding_k*).
            - ``"onehot"``: one-hot encoding (padded to fixed length, flattened).
            - A **callable** ``(list[str]) -> np.ndarray`` of shape
              ``(n_sequences, feature_dim)`` for custom encoders (e.g.,
              pretrained RNA embeddings).

        encoding_k: k-mer sizes when ``encoding="kmer"``. Defaults to [1, 2, 3].

    Example::

        from oligogym.target_features import TargetContextEncoder

        encoder = TargetContextEncoder(
            reference_dir="data/reference_transcripts",
            context_window=50,
            pooling="mean",
            encoding="kmer",
            encoding_k=[1, 2, 3],
        )

        X_ctx_train = encoder.fit_transform(X_train, targets_train)
        X_ctx_test  = encoder.transform(X_test, targets_test)

        # Combine with ASO features
        X_combined = pd.concat([X_aso_train, X_ctx_train], axis=1)
    """

    def __init__(
        self,
        reference_dir: str = "data/reference_transcripts",
        context_window: int = 50,
        pooling: str = "mean",
        encoding: Union[str, callable] = "kmer",
        encoding_k: Optional[list] = None,
    ):
        assert pooling in ("mean", "max", "mean_max"), (
            f"pooling must be 'mean', 'max', or 'mean_max', got '{pooling}'"
        )
        if isinstance(encoding, str):
            assert encoding in ("kmer", "onehot"), (
                f"encoding must be 'kmer', 'onehot', or a callable, got '{encoding}'"
            )

        self.reference_dir = _resolve_reference_dir(reference_dir)
        self.context_window = context_window
        self.pooling = pooling
        self.encoding = encoding
        self.encoding_k = encoding_k or [1, 2, 3]

        self._gene_cache: dict[str, Optional[TranscriptInfo]] = {}
        self._feature_dim: Optional[int] = None
        self._max_context_len: Optional[int] = None  # for onehot
        self._kmer_columns: Optional[list[str]] = None  # for kmer
        self._output_columns: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_gene(self, gene_name: str) -> Optional[TranscriptInfo]:
        """Load and cache transcript info for a gene."""
        if gene_name not in self._gene_cache:
            fasta_path = os.path.join(
                self.reference_dir, f"{gene_name}.fna"
            )
            self._gene_cache[gene_name] = build_transcript_info(fasta_path)
        return self._gene_cache[gene_name]

    def _extract_contexts(self, aso_fasta: str, target_gene: str) -> list:
        """Find all exact binding sites and extract context windows."""
        if (
            target_gene is None
            or target_gene == "negative_control"
            or pd.isna(target_gene)
        ):
            return []

        info = self._load_gene(target_gene)
        if info is None:
            return []

        rc = reverse_complement(aso_fasta)
        exact_hits = find_exact_matches(rc, info.all_transcripts)
        if not exact_hits:
            return []

        contexts = []
        aso_len = len(aso_fasta)
        for acc, positions in exact_hits.items():
            seq = info.all_transcripts[acc]
            for pos in positions:
                start = max(0, pos - self.context_window)
                end = min(len(seq), pos + aso_len + self.context_window)
                contexts.append(seq[start:end])
        return contexts

    def _encode_kmer(self, sequences: list) -> np.ndarray:
        """Encode DNA sequences as k-mer count vectors."""
        col_to_idx = {col: i for i, col in enumerate(self._kmer_columns)}
        result = np.zeros(
            (len(sequences), len(self._kmer_columns)), dtype=np.float32
        )
        for i, seq in enumerate(sequences):
            for k in self.encoding_k:
                for j in range(len(seq) - k + 1):
                    kmer = seq[j : j + k]
                    if kmer in col_to_idx:
                        result[i, col_to_idx[kmer]] += 1
        return result

    def _encode_onehot(self, sequences: list) -> np.ndarray:
        """One-hot encode DNA sequences, pad to fixed length, and flatten."""
        nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        max_len = self._max_context_len
        result = np.zeros((len(sequences), max_len * 4), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for j, nuc in enumerate(seq[:max_len]):
                if nuc in nuc_map:
                    result[i, j * 4 + nuc_map[nuc]] = 1.0
        return result

    def _encode_sequences(self, sequences: list) -> np.ndarray:
        """Dispatch to the configured encoding method."""
        if self.encoding == "kmer":
            return self._encode_kmer(sequences)
        elif self.encoding == "onehot":
            return self._encode_onehot(sequences)
        else:
            return np.asarray(self.encoding(sequences))

    def _pool(self, encoded: np.ndarray) -> np.ndarray:
        """Pool across multiple encoded contexts into a single vector."""
        if self.pooling == "mean":
            return encoded.mean(axis=0)
        elif self.pooling == "max":
            return encoded.max(axis=0)
        else:  # mean_max
            return np.concatenate([encoded.mean(axis=0), encoded.max(axis=0)])

    def _build_column_names(self) -> list:
        """Build descriptive column names based on encoding and pooling."""
        if self.encoding == "kmer":
            base_names = self._kmer_columns
        else:
            base_names = [str(i) for i in range(self._feature_dim)]

        if self.pooling == "mean_max":
            half = len(base_names)
            base_names_half = base_names[:half] if len(base_names) >= half else base_names
            return (
                [f"ctx_mean_{n}" for n in base_names_half]
                + [f"ctx_max_{n}" for n in base_names_half]
            )
        return [f"ctx_{self.pooling}_{n}" for n in base_names]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        oligo_list: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        """Find binding sites, encode contexts, and pool into feature vectors.

        Builds the internal gene cache and determines encoding dimensions.

        Args:
            oligo_list: HELM strings (list or numpy array).
            targets: Target gene names, parallel to oligo_list.

        Returns:
            pd.DataFrame with one row per oligo and pooled context features.
        """
        if isinstance(oligo_list, np.ndarray):
            oligo_list = oligo_list.tolist()
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()
        assert len(oligo_list) == len(targets), (
            f"oligo_list ({len(oligo_list)}) and targets ({len(targets)}) "
            "must have same length"
        )

        # Reset state
        self._gene_cache = {}
        self._feature_dim = None
        self._max_context_len = None
        self._kmer_columns = None

        # Pre-load gene transcripts
        unique_genes = set(
            t for t in targets
            if t is not None and not pd.isna(t) and t != "negative_control"
        )
        genes_found, genes_missing = 0, 0
        for gene in sorted(unique_genes):
            info = self._load_gene(gene)
            if info is not None:
                genes_found += 1
            else:
                genes_missing += 1

        # Extract all context sequences
        all_contexts = []
        for helm, target in zip(oligo_list, targets):
            aso_fasta = _helm_to_fasta(helm)
            all_contexts.append(self._extract_contexts(aso_fasta, target))

        n_matched = sum(1 for c in all_contexts if c)
        n_total = len(all_contexts)

        # Flatten all context sequences for batch encoding
        all_seqs = []
        seq_to_aso = []  # maps flat index -> ASO index
        for i, contexts in enumerate(all_contexts):
            for ctx in contexts:
                all_seqs.append(ctx)
                seq_to_aso.append(i)

        # Initialize encoding parameters
        if self.encoding == "kmer":
            columns = []
            for k in self.encoding_k:
                for kmer in itertools_product("ACGT", repeat=k):
                    columns.append("".join(kmer))
            self._kmer_columns = columns

        if self.encoding == "onehot":
            if all_seqs:
                self._max_context_len = max(len(s) for s in all_seqs)
            else:
                self._max_context_len = self.context_window * 2 + 20

        # Batch encode all sequences at once, then split by ASO and pool
        rows = [None] * len(all_contexts)
        if all_seqs:
            all_encoded = self._encode_sequences(all_seqs)
            aso_groups = defaultdict(list)
            for j, aso_idx in enumerate(seq_to_aso):
                aso_groups[aso_idx].append(all_encoded[j])
            for aso_idx, vecs in aso_groups.items():
                rows[aso_idx] = self._pool(np.array(vecs))

        # Determine feature dimension
        for row in rows:
            if row is not None:
                self._feature_dim = len(row)
                break

        if self._feature_dim is None:
            warnings.warn(
                "TargetContextEncoder: no binding sites found for any ASO. "
                "All context features will be zero."
            )
            if self.encoding == "kmer":
                base_dim = len(self._kmer_columns)
            elif self.encoding == "onehot":
                base_dim = self._max_context_len * 4
            else:
                base_dim = 1
            self._feature_dim = (
                base_dim * 2 if self.pooling == "mean_max" else base_dim
            )

        # Assemble result matrix
        result = np.zeros((len(rows), self._feature_dim), dtype=np.float32)
        for i, row in enumerate(rows):
            if row is not None:
                result[i] = row

        self._output_columns = self._build_column_names()

        enc_label = (
            self.encoding if isinstance(self.encoding, str)
            else self.encoding.__name__
            if hasattr(self.encoding, "__name__")
            else "callable"
        )
        print(
            f"TargetContextEncoder: {len(unique_genes)} gene(s) "
            f"({genes_found} with .fna, {genes_missing} without)"
        )
        print(
            f"  Encoding: {enc_label}, Pooling: {self.pooling}, "
            f"Context window: ±{self.context_window}bp"
        )
        print(
            f"  Matched: {n_matched}/{n_total} ASOs "
            f"({100 * n_matched / max(n_total, 1):.1f}%), "
            f"Feature dim: {self._feature_dim}"
        )

        return pd.DataFrame(result, columns=self._output_columns)

    def transform(
        self,
        oligo_list: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        """Encode contexts using parameters learned during fit_transform.

        New genes not seen during fit_transform are loaded on-the-fly.

        Args:
            oligo_list: HELM strings.
            targets: Target gene names, parallel to oligo_list.

        Returns:
            pd.DataFrame with same columns as fit_transform output.
        """
        if isinstance(oligo_list, np.ndarray):
            oligo_list = oligo_list.tolist()
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()
        assert len(oligo_list) == len(targets), (
            f"oligo_list ({len(oligo_list)}) and targets ({len(targets)}) "
            "must have same length"
        )
        assert self._feature_dim is not None, (
            "Must call fit_transform before transform."
        )

        all_contexts = []
        for helm, target in zip(oligo_list, targets):
            aso_fasta = _helm_to_fasta(helm)
            all_contexts.append(self._extract_contexts(aso_fasta, target))

        # Batch encode all sequences at once
        all_seqs = []
        seq_to_aso = []
        for i, contexts in enumerate(all_contexts):
            for ctx in contexts:
                all_seqs.append(ctx)
                seq_to_aso.append(i)

        rows = [None] * len(all_contexts)
        if all_seqs:
            all_encoded = self._encode_sequences(all_seqs)
            aso_groups = defaultdict(list)
            for j, aso_idx in enumerate(seq_to_aso):
                aso_groups[aso_idx].append(all_encoded[j])
            for aso_idx, vecs in aso_groups.items():
                rows[aso_idx] = self._pool(np.array(vecs))

        result = np.zeros((len(rows), self._feature_dim), dtype=np.float32)
        for i, row in enumerate(rows):
            if row is not None:
                result[i] = row

        return pd.DataFrame(result, columns=self._output_columns)
