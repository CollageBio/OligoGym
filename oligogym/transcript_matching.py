"""
Transcript Matching Utilities
=============================
General-purpose utilities for matching oligonucleotide sequences to their
binding sites on target mRNA transcripts. Handles FASTA parsing, reverse
complement computation, exact and fuzzy substring matching, ORF detection,
and per-gene transcript caching.

These utilities are gene-agnostic: they work with any gene for which a
reference transcript FASTA file ({GENE}.fna) is available
in the reference directory.
"""

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

# =============================================================================
# Data Structures
# =============================================================================

COMPLEMENT = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")
STOP_CODONS = {"TAA", "TAG", "TGA"}
DINUCLEOTIDES = [a + b for a in "ACGT" for b in "ACGT"]


@dataclass
class TranscriptInfo:
    """Cached information about a gene's reference transcripts."""

    accession: str  # canonical transcript accession (e.g., "NM_005910.6")
    sequence: str  # full transcript DNA sequence (uppercase)
    length: int  # len(sequence)
    gc_content: float  # GC fraction of full transcript
    cds_start: Optional[int]  # 0-indexed CDS start (None if no ORF)
    cds_end: Optional[int]  # 0-indexed CDS end, exclusive
    utr5_length: Optional[int]
    utr3_length: Optional[int]
    cds_length: Optional[int]
    all_transcripts: dict = field(repr=False)  # {accession: sequence}
    num_nm_transcripts: int = 0


@dataclass
class MatchResult:
    """Result of matching one ASO to a gene's transcripts."""

    matched: bool
    match_pos: Optional[int]  # 0-indexed position on canonical transcript
    num_isoforms_matched: int
    total_isoforms: int
    match_type: str  # "exact", "fuzzy", or "none"


# =============================================================================
# Sequence Operations
# =============================================================================


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    """Calculate GC fraction of a DNA sequence."""
    if len(seq) == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def dinucleotide_frequencies(seq: str) -> dict:
    """Calculate frequency of each of the 16 dinucleotides in a DNA sequence."""
    total = max(len(seq) - 1, 1)
    counts = Counter(seq[i : i + 2] for i in range(len(seq) - 1))
    return {d: counts.get(d, 0) / total for d in DINUCLEOTIDES}


def shannon_entropy(seq: str) -> float:
    """Calculate Shannon entropy of nucleotide frequencies in a DNA sequence."""
    if len(seq) == 0:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def max_homopolymer_length(seq: str) -> int:
    """Find the length of the longest single-nucleotide run in a sequence."""
    if len(seq) == 0:
        return 0
    max_len = 1
    current_len = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_len += 1
            if current_len > max_len:
                max_len = current_len
        else:
            current_len = 1
    return max_len


# =============================================================================
# FASTA Parsing
# =============================================================================


def parse_fasta(fasta_path: str, prefix_filter: str = "NM_") -> dict:
    """Parse a FASTA file, optionally filtering by accession prefix.

    Args:
        fasta_path: Path to the .fna FASTA file.
        prefix_filter: Only keep entries whose accession starts with this.
            Set to None to keep all entries.

    Returns:
        Dictionary mapping accession → uppercase DNA sequence.
    """
    transcripts = {}
    current_acc = None
    current_seq_parts = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_acc is not None:
                    transcripts[current_acc] = "".join(current_seq_parts).upper()
                acc = line[1:].split()[0]
                if prefix_filter is None or acc.startswith(prefix_filter):
                    current_acc = acc
                    current_seq_parts = []
                else:
                    current_acc = None
                    current_seq_parts = []
            else:
                if current_acc is not None:
                    current_seq_parts.append(line)
        if current_acc is not None:
            transcripts[current_acc] = "".join(current_seq_parts).upper()

    return transcripts


def select_canonical(transcripts: dict) -> Optional[str]:
    """Select the canonical transcript as the longest NM_ transcript.

    Returns:
        Accession string of the longest transcript, or None if empty.
    """
    if not transcripts:
        return None
    return max(transcripts.keys(), key=lambda acc: len(transcripts[acc]))


# =============================================================================
# ORF Detection
# =============================================================================


def find_longest_orf(sequence: str) -> Optional[tuple]:
    """Find the longest open reading frame in a transcript sequence.

    Scans every position for ATG start codons, then reads in-frame until
    a stop codon (TAA/TAG/TGA). Returns the longest such ORF.

    For RefSeq NM_ mRNA transcripts (5'UTR + CDS + 3'UTR), the longest
    ORF reliably corresponds to the annotated CDS.

    Args:
        sequence: Uppercase DNA sequence of a transcript.

    Returns:
        (cds_start, cds_end) as 0-indexed positions where cds_end is exclusive,
        so sequence[cds_start:cds_end] gives the CDS including the stop codon.
        Returns None if no ORF is found.
    """
    best_orf = None
    best_length = 0
    seq_len = len(sequence)

    for i in range(seq_len - 2):
        if sequence[i : i + 3] == "ATG":
            j = i + 3
            while j <= seq_len - 3:
                codon = sequence[j : j + 3]
                if codon in STOP_CODONS:
                    orf_length = (j + 3) - i
                    if orf_length > best_length:
                        best_length = orf_length
                        best_orf = (i, j + 3)
                    break
                j += 3

    return best_orf


# =============================================================================
# Substring Matching
# =============================================================================


def find_exact_matches(query: str, transcripts: dict) -> dict:
    """Search for query as an exact substring in each transcript.

    Args:
        query: DNA sequence to search for (e.g., reverse complement of ASO).
        transcripts: {accession: sequence} dictionary.

    Returns:
        Dictionary {accession: [list of 0-indexed start positions]}.
        Only accessions with at least one match are included.
    """
    hits = {}
    for acc, tseq in transcripts.items():
        positions = []
        start = 0
        while True:
            idx = tseq.find(query, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1
        if positions:
            hits[acc] = positions
    return hits


def hamming_search(query: str, target: str, max_mismatches: int = 2) -> list:
    """Slide query along target, returning positions with ≤ max_mismatches.

    Only returns positions with at least 1 mismatch (exact matches excluded).

    Args:
        query: Short DNA sequence to match.
        target: Longer DNA sequence to search within.
        max_mismatches: Maximum Hamming distance allowed.

    Returns:
        List of (start_position, num_mismatches) tuples.
    """
    qlen = len(query)
    tlen = len(target)
    hits = []
    for i in range(tlen - qlen + 1):
        mm = sum(1 for a, b in zip(query, target[i : i + qlen]) if a != b)
        if 0 < mm <= max_mismatches:
            hits.append((i, mm))
    return hits


# =============================================================================
# High-Level Matching
# =============================================================================


def match_aso_to_transcripts(
    aso_fasta: str,
    transcript_info: TranscriptInfo,
) -> MatchResult:
    """Match a single ASO to a gene's transcripts via exact matching.

    Searches for the reverse complement of the ASO as an exact substring
    across all NM_ isoforms. Reports the position on the canonical
    transcript when available, otherwise the first isoform hit.

    Args:
        aso_fasta: ASO base sequence (DNA alphabet, e.g., "GATTCAGCCTG").
        transcript_info: Pre-parsed transcript information for the target gene.

    Returns:
        MatchResult with match status, position, and isoform counts.
    """
    rc = reverse_complement(aso_fasta)
    all_transcripts = transcript_info.all_transcripts
    canonical_acc = transcript_info.accession

    # Exact matches across all isoforms
    exact_hits = find_exact_matches(rc, all_transcripts)

    if exact_hits:
        if canonical_acc in exact_hits:
            match_pos = exact_hits[canonical_acc][0]
        else:
            first_acc = next(iter(exact_hits))
            match_pos = exact_hits[first_acc][0]

        return MatchResult(
            matched=True,
            match_pos=match_pos,
            num_isoforms_matched=len(exact_hits),
            total_isoforms=len(all_transcripts),
            match_type="exact",
        )

    return MatchResult(
        matched=False,
        match_pos=None,
        num_isoforms_matched=0,
        total_isoforms=len(all_transcripts),
        match_type="none",
    )


# =============================================================================
# Gene Cache Builder
# =============================================================================


def build_transcript_info(fasta_path: str) -> Optional[TranscriptInfo]:
    """Parse a gene's FASTA file and build a TranscriptInfo object.

    Prefers NM_ (coding mRNA) transcripts. If none are found, falls back
    to NR_ (non-coding RNA) transcripts — this handles lncRNAs like MALAT1
    and snoRNA host genes like SNHG14.

    Args:
        fasta_path: Path to {GENE}.fna file.

    Returns:
        TranscriptInfo object, or None if file doesn't exist or has no
        NM_/NR_ entries.
    """
    if not os.path.exists(fasta_path):
        return None

    # Try NM_ first (coding mRNA), fall back to NR_ (non-coding RNA)
    all_transcripts = parse_fasta(fasta_path, prefix_filter="NM_")
    if not all_transcripts:
        all_transcripts = parse_fasta(fasta_path, prefix_filter="NR_")
    if not all_transcripts:
        return None

    canonical_acc = select_canonical(all_transcripts)
    canonical_seq = all_transcripts[canonical_acc]

    orf = find_longest_orf(canonical_seq)
    cds_start = orf[0] if orf else None
    cds_end = orf[1] if orf else None

    return TranscriptInfo(
        accession=canonical_acc,
        sequence=canonical_seq,
        length=len(canonical_seq),
        gc_content=gc_content(canonical_seq),
        cds_start=cds_start,
        cds_end=cds_end,
        utr5_length=cds_start if cds_start is not None else None,
        utr3_length=(len(canonical_seq) - cds_end) if cds_end is not None else None,
        cds_length=(cds_end - cds_start)
        if (cds_start is not None and cds_end is not None)
        else None,
        all_transcripts=all_transcripts,
        num_nm_transcripts=len(all_transcripts),
    )
