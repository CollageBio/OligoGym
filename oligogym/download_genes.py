#!/usr/bin/env python3
"""
Download gene transcript sequences from NCBI using Biopython Entrez.

Given gene symbols (e.g., MAPT, APP), this script:
1. Resolves gene symbols to NCBI Gene IDs via Entrez esearch
2. Links Gene IDs to RefSeq RNA transcript accessions via Entrez elink
3. Downloads FASTA sequences via Entrez efetch
4. Saves per-gene FASTA files ({GENE}.fna)

Usage:
    python download_genes.py --email you@example.com MAPT
    python download_genes.py --email you@example.com MAPT APP APOE
    python download_genes.py --email you@example.com --genes MAPT APP PSEN1
    python download_genes.py --email you@example.com --genes-file genes.txt
    python download_genes.py --email you@example.com --api-key YOUR_KEY MAPT
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from Bio import Entrez


# Map common taxon names to scientific names (for Entrez search terms)
TAXON_MAP = {
    "human": "Homo sapiens",
    "mouse": "Mus musculus",
    "rat": "Rattus norvegicus",
}


class NCBIGeneDownloader:
    """Download gene transcript data from NCBI using Entrez."""

    def __init__(
        self,
        email: str,
        output_dir: str = "gene_downloads",
        api_key: Optional[str] = None,
    ):
        """Initialize downloader.

        Args:
            email: Email address (required by NCBI for Entrez usage).
            output_dir: Directory to save downloaded FASTA files.
            api_key: Optional NCBI API key (raises rate limit to 10 req/s).
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def resolve_gene_symbol(
        self, gene_symbol: str, taxon: str = "human"
    ) -> Optional[int]:
        """Resolve a gene symbol to its NCBI Gene ID.

        Args:
            gene_symbol: Gene symbol (e.g., 'MAPT', 'APP').
            taxon: Organism name ('human', 'mouse', 'rat') or scientific name.

        Returns:
            NCBI Gene ID, or None if not found.
        """
        organism = TAXON_MAP.get(taxon.lower(), taxon)
        term = f"{gene_symbol}[Gene Name] AND {organism}[Organism]"

        try:
            handle = Entrez.esearch(db="gene", term=term, retmax=1)
            record = Entrez.read(handle)
            handle.close()

            if record["IdList"]:
                gene_id = int(record["IdList"][0])
                print(f"  Resolved {gene_symbol} -> Gene ID: {gene_id}")
                return gene_id
            else:
                print(f"  Could not resolve gene symbol: {gene_symbol}")
                return None

        except Exception as e:
            print(f"  Error resolving {gene_symbol}: {e}")
            return None

    def download_gene_transcripts(
        self, gene_id: int, gene_symbol: str
    ) -> Optional[Path]:
        """Download RefSeq RNA transcripts for a gene as FASTA.

        Uses Entrez elink to find RefSeq RNA records linked to the Gene ID,
        then efetch to download them in FASTA format.

        Args:
            gene_id: NCBI Gene ID.
            gene_symbol: Gene symbol (used for the output filename).

        Returns:
            Path to the saved FASTA file, or None if failed.
        """
        try:
            # Link Gene ID -> RefSeq RNA transcript accessions
            handle = Entrez.elink(
                dbfrom="gene",
                db="nuccore",
                id=str(gene_id),
                linkname="gene_nuccore_refseqrna",
            )
            link_results = Entrez.read(handle)
            handle.close()

            # Extract linked nucleotide IDs
            transcript_ids = []
            for linkset in link_results:
                if "LinkSetDb" in linkset:
                    for db_link in linkset["LinkSetDb"]:
                        for link in db_link["Link"]:
                            transcript_ids.append(link["Id"])

            if not transcript_ids:
                print(f"  No RefSeq RNA transcripts found for Gene ID {gene_id}")
                return None

            print(f"  Found {len(transcript_ids)} transcript(s), downloading FASTA...")

            # Fetch FASTA sequences for all transcripts
            handle = Entrez.efetch(
                db="nuccore",
                id=transcript_ids,
                rettype="fasta",
                retmode="text",
            )
            fasta_data = handle.read()
            handle.close()

            # Save to file
            output_path = self.output_dir / f"{gene_symbol}.fna"
            with open(output_path, "w") as f:
                f.write(fasta_data)

            print(f"  Saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"  Error downloading transcripts for {gene_symbol}: {e}")
            return None

    def process_genes(
        self,
        gene_symbols: List[str],
        taxon: str = "human",
        delay: float = 0.5,
    ) -> Dict[str, Optional[Path]]:
        """Process multiple gene symbols: resolve and download transcripts.

        Args:
            gene_symbols: List of gene symbols to process.
            taxon: Organism name or scientific name.
            delay: Delay between genes (seconds) to respect NCBI rate limits.

        Returns:
            Dictionary mapping gene symbols to output file paths (None if failed).
        """
        results = {}

        for gene_symbol in gene_symbols:
            print(f"\n{'=' * 60}")
            print(f"Processing: {gene_symbol}")
            print(f"{'=' * 60}")

            gene_id = self.resolve_gene_symbol(gene_symbol, taxon)

            if gene_id:
                output_path = self.download_gene_transcripts(gene_id, gene_symbol)
                results[gene_symbol] = output_path
            else:
                results[gene_symbol] = None

            time.sleep(delay)

        return results

    def summarize_results(self, results: Dict[str, Optional[Path]]):
        """Print summary of download results."""
        print(f"\n\n{'=' * 60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'=' * 60}")

        successful = [g for g, p in results.items() if p is not None]
        failed = [g for g, p in results.items() if p is None]

        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        for gene in successful:
            print(f"  + {gene}: {results[gene]}")

        if failed:
            print(f"\nFailed: {len(failed)}/{len(results)}")
            for gene in failed:
                print(f"  - {gene}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download gene transcript FASTA files from NCBI via Entrez",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --email you@example.com MAPT
  %(prog)s --email you@example.com MAPT APP APOE PSEN1
  %(prog)s --email you@example.com --genes MAPT APP APOE
  %(prog)s --email you@example.com --genes-file genes.txt
  %(prog)s --email you@example.com --api-key YOUR_KEY MAPT --taxon mouse
        """,
    )

    parser.add_argument(
        "genes", nargs="*", help="Gene symbol(s) to download (e.g., MAPT APP APOE)"
    )
    parser.add_argument(
        "--genes",
        dest="genes_list",
        nargs="+",
        help="Alternative way to specify gene symbols",
    )
    parser.add_argument(
        "--genes-file",
        type=str,
        help="Path to file containing gene symbols (one per line)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=os.environ.get("NCBI_EMAIL"),
        help="Email address (required by NCBI Entrez). "
        "Falls back to NCBI_EMAIL env var if not provided.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NCBI_API_KEY"),
        help="NCBI API key (optional, raises rate limit to 10 req/s). "
        "Falls back to NCBI_API_KEY env var if not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gene_transcripts",
        help="Output directory for downloaded FASTA files (default: gene_transcripts)",
    )
    parser.add_argument(
        "--taxon",
        type=str,
        default="human",
        help="Organism (human, mouse, rat, or scientific name) (default: human)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds (default: 0.5)",
    )

    return parser.parse_args()


def read_genes_from_file(filepath: str) -> List[str]:
    """Read gene symbols from a file (one per line).

    Args:
        filepath: Path to file containing gene symbols.

    Returns:
        List of gene symbols.
    """
    genes = []
    with open(filepath, "r") as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                genes.append(gene)
    return genes


def main():
    """Main execution function."""
    args = parse_arguments()

    # Collect gene symbols from various sources
    target_genes = []

    if args.genes:
        target_genes.extend(args.genes)
    if args.genes_list:
        target_genes.extend(args.genes_list)
    if args.genes_file:
        try:
            target_genes.extend(read_genes_from_file(args.genes_file))
        except FileNotFoundError:
            print(f"Error: Gene file '{args.genes_file}' not found")
            return
        except Exception as e:
            print(f"Error reading gene file: {e}")
            return

    if not args.email:
        print("Error: Email is required by NCBI Entrez.")
        print("Provide via --email or set the NCBI_EMAIL environment variable.")
        return

    if not target_genes:
        print("Error: No gene symbols provided")
        print("Use: python download_genes.py --email you@example.com MAPT APP")
        return

    # Remove duplicates while preserving order
    target_genes = list(dict.fromkeys(target_genes))

    print(f"{'=' * 60}")
    print("NCBI Gene Transcript Downloader (Entrez)")
    print(f"{'=' * 60}")
    print(f"Genes to process: {', '.join(target_genes)}")
    print(f"Organism: {args.taxon}")
    print(f"Output directory: {args.output_dir}")
    if args.api_key:
        print("API key: provided")
    print(f"{'=' * 60}\n")

    downloader = NCBIGeneDownloader(
        email=args.email,
        output_dir=args.output_dir,
        api_key=args.api_key,
    )

    results = downloader.process_genes(
        gene_symbols=target_genes,
        taxon=args.taxon,
        delay=args.delay,
    )

    downloader.summarize_results(results)


if __name__ == "__main__":
    main()
