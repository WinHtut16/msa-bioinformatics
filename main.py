"""
main.py
--------------------
Main entry point for the MSA Project.

Runs the full pipeline:
    1. Generate or load sequences
    2. Run Progressive MSA (proposed algorithm)
    3. Run Star Alignment (baseline)
    4. Compare results
    5. Run experiments and generate plots

Usage:
    python main.py                  # run demo with synthetic data
    python main.py --experiments    # run full experiment suite + plots
    python main.py --fasta FILE     # align sequences from a FASTA file

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import os
import sys
import argparse
import time

# Add src/ to path so all modules are importable
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)

from data_generator  import generate_sequences
from progressive_msa import progressive_msa, validate_alignment
from star_alignment  import star_alignment
import colorama
colorama.init()
from output_formatter import (
    print_demo_header,
    print_fasta_header,
    print_alignment_result,
    print_comparison,
    print_legend,
    print_stats_summary,      # NEW — Tier 1 improvement
)


# ─────────────────────────────────────────────
# 1.  FASTA loader (simple, no Biopython needed)
# ─────────────────────────────────────────────

def load_fasta(filepath: str) -> tuple:
    """
    Load sequences from a FASTA file.

    Parameters
    ----------
    filepath : str — path to .fasta or .fa file

    Returns
    -------
    (sequences, labels) : tuple of lists
    """
    sequences, labels = [], []
    current_label, current_seq = None, []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_label is not None:
                    sequences.append(''.join(current_seq).upper())
                    labels.append(current_label)
                current_label = line[1:].split()[0]   # take first word
                current_seq   = []
            else:
                current_seq.append(line)

    if current_label is not None:
        sequences.append(''.join(current_seq).upper())
        labels.append(current_label)

    return sequences, labels


# ─────────────────────────────────────────────
# 2.  Demo mode
# ─────────────────────────────────────────────

def run_demo(show_stats: bool = False):
    """
    Run a demonstration on synthetic sequences.
    Shows the full pipeline with clear output.

    Parameters
    ----------
    show_stats : bool
        If True, print the detailed alignment statistics panel after
        each alignment result.
    """
    m, l = 6, 40
    print_demo_header(m=m, l=l, mutation_rate=0.15, seed=42)

    # Generate synthetic sequences
    print("\n[Generating synthetic DNA sequences ...]")
    seqs, labels = generate_sequences(
        m=m, l=l,
        mutation_rate=0.15,
        indel_rate=0.05,
        seed=42
    )
    print(f"  Generated {m} sequences of base length {l}")
    print("\n  Raw (unaligned) sequences:")
    max_lbl = max(len(lb) for lb in labels)
    for lbl, seq in zip(labels, seqs):
        print(f"    {lbl:<{max_lbl}} (len={len(seq):3d}) : {seq}")

    # ── Run Progressive MSA ───────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  Running Progressive MSA ...")
    prog_result = progressive_msa(
        seqs, labels,
        gap_open=-2.0, gap_extend=-0.5,
        verbose=True
    )

    valid = validate_alignment(prog_result['aligned_sequences'], seqs)
    print(f"\n  Alignment validation : {'PASSED ✓' if valid else 'FAILED ✗'}")

    print_alignment_result(
        prog_result['aligned_sequences'],
        labels,
        prog_result['sp_score'],
        prog_result['runtime'],
        "Progressive MSA",
        show_stats=not show_stats 
    )
    print_legend()

    # Detailed stats panel — only when --stats is passed
    if show_stats:
        print_stats_summary(
            prog_result['aligned_sequences'],
            labels,
            algorithm_name="Progressive MSA",
        )

    # ── Run Star Alignment ────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  Running Star Alignment (baseline) ...")
    star_result = star_alignment(
        seqs, labels,
        gap_open=-2.0, gap_extend=-0.5,
        verbose=True
    )

    print_alignment_result(
        star_result['aligned_sequences'],
        labels,
        star_result['sp_score'],
        star_result['runtime'],
        "Star Alignment",
        show_stats=not show_stats 
    )
    print_legend()

    if show_stats:
        print_stats_summary(
            star_result['aligned_sequences'],
            labels,
            algorithm_name="Star Alignment",
        )

    # ── Comparison ────────────────────────────────────────────────────────
    print_comparison(
        prog_result['aligned_sequences'], labels,
        prog_result['sp_score'], prog_result['runtime'],
        star_result['aligned_sequences'], labels,
        star_result['sp_score'], star_result['runtime']
    )


# ─────────────────────────────────────────────
# 3.  Experiment mode
# ─────────────────────────────────────────────

def run_experiments():
    """
    Run the full experiment suite and generate plots.
    """
    print("\n" + "█"*65)
    print("  PROGRESSIVE MSA — Experiment Mode")
    print("█"*65)

    from experiments import (experiment_runtime_vs_m,
                              experiment_runtime_vs_l,
                              experiment_quality_vs_mutation,
                              plot_all_results)

    exp1 = experiment_runtime_vs_m(
        m_values = [2, 4, 6, 8, 10, 12, 14],
        l        = 50,
        mut_rate = 0.1,
        n_runs   = 3,
        seed     = 42
    )
    exp2 = experiment_runtime_vs_l(
        l_values = [20, 40, 60, 80, 100, 120, 150],
        m        = 6,
        mut_rate = 0.1,
        n_runs   = 3,
        seed     = 42
    )
    exp3 = experiment_quality_vs_mutation(
        mut_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        m         = 8,
        l         = 60,
        n_runs    = 3,
        seed      = 42
    )

    print("\nGenerating plots ...")
    out_path = plot_all_results(exp1, exp2, exp3)

    print(f"\n  Experiments complete.")
    print(f"  Plots saved to : {os.path.abspath(out_path)}")


# ─────────────────────────────────────────────
# 4.  FASTA mode
# ─────────────────────────────────────────────

def run_fasta(filepath: str, show_stats: bool = False, output: str = None):
    """
    Align sequences loaded from a FASTA file.

    Parameters
    ----------
    filepath   : str  — path to a .fasta / .fa file
    show_stats : bool — if True, print detailed statistics panels
    """
    if not os.path.exists(filepath):
        print(f"\n  ERROR: File not found: {filepath}")
        sys.exit(1)

    seqs, labels = load_fasta(filepath)
    print_fasta_header(filepath, len(seqs))

    if len(seqs) < 2:
        print("  ERROR: Need at least 2 sequences for alignment.")
        sys.exit(1)

    max_lbl = max(len(lb) for lb in labels) if labels else 0
    for lbl, seq in zip(labels, seqs):
        print(f"  {lbl:<{max_lbl}} (len={len(seq):4d})")

    # Run both algorithms
    print("\n[Running Progressive MSA ...]")
    prog_result = progressive_msa(seqs, labels, verbose=False)

    print("[Running Star Alignment ...]")
    star_result = star_alignment(seqs, labels, verbose=False)

    # ── Progressive result ────────────────────────────────────────────────
    print_alignment_result(
        prog_result['aligned_sequences'],
        labels,
        prog_result['sp_score'],
        prog_result['runtime'],
        "Progressive MSA",
        show_stats=not show_stats 
    )
    print_legend()

    if show_stats:
        print_stats_summary(
            prog_result['aligned_sequences'],
            labels,
            algorithm_name="Progressive MSA",
        )

    # ── Star result ───────────────────────────────────────────────────────
    print_alignment_result(
        star_result['aligned_sequences'],
        labels,
        star_result['sp_score'],
        star_result['runtime'],
        "Star Alignment",
        show_stats=not show_stats 
            
    )
    print_legend()

    if show_stats:
        print_stats_summary(
            star_result['aligned_sequences'],
            labels,
            algorithm_name="Star Alignment",
        )

    # ── Comparison ────────────────────────────────────────────────────────
    print_comparison(
        prog_result['aligned_sequences'], labels,
        prog_result['sp_score'], prog_result['runtime'],
        star_result['aligned_sequences'], labels,
        star_result['sp_score'], star_result['runtime']
    )

    # ── Save progressive alignment to FASTA ──────────────────────────────
    if output:
        if os.path.isdir(output):
            raise ValueError("Output must be a file path, not a directory")
        out_file = output
    else:
        base, ext = os.path.splitext(filepath)
        out_file = f"{base}_aligned{ext}"
    with open(out_file, 'w') as f:
        for lbl, seq in zip(labels, prog_result['aligned_sequences']):
            f.write(f">{lbl}\n{seq}\n")
    print(f"\n  Progressive alignment saved to: {out_file}")


# ─────────────────────────────────────────────
# 5.  Argument parsing & entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Efficient Multiple Sequence Alignment\n"
            "Progressive MSA vs Star Alignment — AIT 2026"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                          # demo on synthetic sequences\n"
            "  python main.py --stats                  # demo + detailed stats panel\n"
            "  python main.py --fasta data/seqs.fasta  # align a FASTA file\n"
            "  python main.py --fasta data/seqs.fasta --stats  # with stats\n"
            "  python main.py --experiments            # full benchmark suite\n"
        )
    )
    parser.add_argument(
        '--experiments', action='store_true',
        help='Run full experiment suite and generate plots'
    )
    parser.add_argument(
        '--fasta', type=str, default=None,
        metavar='FILE',
        help='Path to a FASTA file to align'
    )
    parser.add_argument(
        '--stats', action='store_true',
        help=(
            'Print detailed alignment statistics after each result:\n'
            '  - Conserved / variable / gap column counts with bar charts\n'
            '  - Residue frequency breakdown (A, C, G, T, gap)\n'
            '  - Column type minimap across the full alignment\n'
            '  - Conserved region list with biological significance labels'
        )
    )
    parser.add_argument(
    '--output', type=str, default=None,
    metavar='FILE',
    help='Save the progressive alignment to this FASTA file (default: input_aligned.fasta)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.fasta:
        run_fasta(args.fasta, show_stats=args.stats, output=args.output)
    elif args.experiments:
        run_experiments()
    else:
        run_demo(show_stats=args.stats)