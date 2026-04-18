"""
experiments.py
--------------------
Module 7b: Experimental Design, Benchmarking & Plotting

Runs systematic experiments comparing:
    - Progressive MSA (proposed algorithm)
    - Star Alignment (baseline)

Experiments:
    1. Runtime vs number of sequences (m)  — fixed l
    2. Runtime vs sequence length (l)      — fixed m
    3. SP score vs mutation rate           — quality comparison
    4. Runtime vs m (log-log)              — asymptotic validation

Metrics collected:
    - Wall-clock runtime (seconds)
    - Sum-of-Pairs (SP) alignment score
    - Alignment length

All plots saved to results/ directory.

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_generator  import generate_sequences
from progressive_msa import progressive_msa
from star_alignment  import star_alignment

# Ensure results directory exists
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  Single Run Helper
# ─────────────────────────────────────────────

def run_once(algorithm_fn, sequences, labels,
             gap_open=-2.0, gap_extend=-0.5) -> dict:
    """
    Run one algorithm on one dataset and return metrics.
    """
    result = algorithm_fn(
        sequences, labels,
        gap_open=gap_open, gap_extend=gap_extend,
        verbose=False
    )
    return {
        'runtime'  : result['runtime'],
        'sp_score' : result['sp_score'],
        'aln_len'  : len(result['aligned_sequences'][0]),
    }


def average_runs(algorithm_fn, sequences, labels,
                 n_runs=3, gap_open=-2.0, gap_extend=-0.5) -> dict:
    """
    Run algorithm n_runs times and return mean ± std of metrics.
    Uses fixed seeds for reproducibility.
    """
    runtimes  = []
    sp_scores = []
    for _ in range(n_runs):
        r = run_once(algorithm_fn, sequences, labels, gap_open, gap_extend)
        runtimes.append(r['runtime'])
        sp_scores.append(r['sp_score'])
    return {
        'runtime_mean' : np.mean(runtimes),
        'runtime_std'  : np.std(runtimes),
        'sp_mean'      : np.mean(sp_scores),
        'sp_std'       : np.std(sp_scores),
        'aln_len'      : r['aln_len'],
    }


# ─────────────────────────────────────────────
# 2.  Experiment 1: Runtime vs m (fixed l)
# ─────────────────────────────────────────────

def experiment_runtime_vs_m(m_values   : list,
                              l          : int   = 50,
                              mut_rate   : float = 0.1,
                              n_runs     : int   = 3,
                              seed       : int   = 42) -> dict:
    """
    Measure runtime as number of sequences m varies (fixed length l).

    Parameters
    ----------
    m_values : list of int — values of m to test
    l        : int — fixed sequence length
    mut_rate : float — mutation rate for sequence generation
    n_runs   : int — runs per configuration
    seed     : int — random seed

    Returns
    -------
    dict with keys 'prog' and 'star', each containing lists of metrics
    """
    print(f"\n{'='*55}")
    print(f"Experiment 1: Runtime vs m  (l={l}, mut={mut_rate})")
    print(f"{'='*55}")

    prog_means, prog_stds = [], []
    star_means, star_stds = [], []

    for m in m_values:
        seqs, labels = generate_sequences(
            m=m, l=l, mutation_rate=mut_rate,
            indel_rate=0.03, seed=seed
        )
        print(f"  m={m:3d}, l={l} ...", end=' ', flush=True)

        p = average_runs(progressive_msa, seqs, labels, n_runs=n_runs)
        s = average_runs(star_alignment,  seqs, labels, n_runs=n_runs)

        prog_means.append(p['runtime_mean'])
        prog_stds.append(p['runtime_std'])
        star_means.append(s['runtime_mean'])
        star_stds.append(s['runtime_std'])

        print(f"Progressive={p['runtime_mean']:.3f}s  "
              f"Star={s['runtime_mean']:.3f}s")

    return {
        'prog': {'means': prog_means, 'stds': prog_stds},
        'star': {'means': star_means, 'stds': star_stds},
        'm_values': m_values,
        'l': l,
    }


# ─────────────────────────────────────────────
# 3.  Experiment 2: Runtime vs l (fixed m)
# ─────────────────────────────────────────────

def experiment_runtime_vs_l(l_values   : list,
                              m          : int   = 6,
                              mut_rate   : float = 0.1,
                              n_runs     : int   = 3,
                              seed       : int   = 42) -> dict:
    """
    Measure runtime as sequence length l varies (fixed number m).
    """
    print(f"\n{'='*55}")
    print(f"Experiment 2: Runtime vs l  (m={m}, mut={mut_rate})")
    print(f"{'='*55}")

    prog_means, prog_stds = [], []
    star_means, star_stds = [], []

    for l in l_values:
        seqs, labels = generate_sequences(
            m=m, l=l, mutation_rate=mut_rate,
            indel_rate=0.03, seed=seed
        )
        print(f"  m={m}, l={l:4d} ...", end=' ', flush=True)

        p = average_runs(progressive_msa, seqs, labels, n_runs=n_runs)
        s = average_runs(star_alignment,  seqs, labels, n_runs=n_runs)

        prog_means.append(p['runtime_mean'])
        prog_stds.append(p['runtime_std'])
        star_means.append(s['runtime_mean'])
        star_stds.append(s['runtime_std'])

        print(f"Progressive={p['runtime_mean']:.3f}s  "
              f"Star={s['runtime_mean']:.3f}s")

    return {
        'prog': {'means': prog_means, 'stds': prog_stds},
        'star': {'means': star_means, 'stds': star_stds},
        'l_values': l_values,
        'm': m,
    }


# ─────────────────────────────────────────────
# 4.  Experiment 3: SP Score vs Mutation Rate
# ─────────────────────────────────────────────

def experiment_quality_vs_mutation(mut_rates  : list,
                                    m          : int = 8,
                                    l          : int = 60,
                                    n_runs     : int = 3,
                                    seed       : int = 42) -> dict:
    """
    Compare SP scores of Progressive vs Star across mutation rates.
    Higher mutation rate = more divergent sequences = harder alignment.
    """
    print(f"\n{'='*55}")
    print(f"Experiment 3: SP Score vs Mutation Rate  (m={m}, l={l})")
    print(f"{'='*55}")

    prog_sp_means, prog_sp_stds = [], []
    star_sp_means, star_sp_stds = [], []

    for rate in mut_rates:
        seqs, labels = generate_sequences(
            m=m, l=l, mutation_rate=rate,
            indel_rate=rate * 0.3, seed=seed
        )
        print(f"  mut_rate={rate:.2f} ...", end=' ', flush=True)

        p = average_runs(progressive_msa, seqs, labels, n_runs=n_runs)
        s = average_runs(star_alignment,  seqs, labels, n_runs=n_runs)

        prog_sp_means.append(p['sp_mean'])
        prog_sp_stds.append(p['sp_std'])
        star_sp_means.append(s['sp_mean'])
        star_sp_stds.append(s['sp_std'])

        print(f"Progressive SP={p['sp_mean']:.2f}  "
              f"Star SP={s['sp_mean']:.2f}")

    return {
        'prog': {'sp_means': prog_sp_means, 'sp_stds': prog_sp_stds},
        'star': {'sp_means': star_sp_means, 'sp_stds': star_sp_stds},
        'mut_rates': mut_rates,
        'm': m, 'l': l,
    }


# ─────────────────────────────────────────────
# 5.  Plotting
# ─────────────────────────────────────────────

def plot_all_results(exp1, exp2, exp3):
    """
    Generate a 2x2 figure with all 4 experiment plots and save to results/.

    Plot layout:
        [0,0] Runtime vs m (linear)
        [0,1] Runtime vs l (linear)
        [1,0] SP Score vs Mutation Rate
        [1,1] Runtime vs m (log-log — asymptotic validation)
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Efficient Multiple Sequence Alignment — Experimental Results\n"
        "Progressive MSA vs Star Alignment Baseline",
        fontsize=13, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.32)

    PROG_COLOR = '#2196F3'   # blue
    STAR_COLOR = '#FF5722'   # orange-red
    ALPHA      = 0.25        # shaded error region transparency

    # ── Plot 1: Runtime vs m ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    m_vals = exp1['m_values']
    ax1.plot(m_vals, exp1['prog']['means'],
             color=PROG_COLOR, marker='o', label='Progressive MSA', lw=2)
    ax1.fill_between(m_vals,
                     np.array(exp1['prog']['means']) - np.array(exp1['prog']['stds']),
                     np.array(exp1['prog']['means']) + np.array(exp1['prog']['stds']),
                     alpha=ALPHA, color=PROG_COLOR)
    ax1.plot(m_vals, exp1['star']['means'],
             color=STAR_COLOR, marker='s', label='Star Alignment', lw=2,
             linestyle='--')
    ax1.fill_between(m_vals,
                     np.array(exp1['star']['means']) - np.array(exp1['star']['stds']),
                     np.array(exp1['star']['means']) + np.array(exp1['star']['stds']),
                     alpha=ALPHA, color=STAR_COLOR)
    ax1.set_xlabel('Number of Sequences (m)', fontsize=11)
    ax1.set_ylabel('Runtime (seconds)',        fontsize=11)
    ax1.set_title(f'Runtime vs m  (l={exp1["l"]})',  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ── Plot 2: Runtime vs l ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    l_vals = exp2['l_values']
    ax2.plot(l_vals, exp2['prog']['means'],
             color=PROG_COLOR, marker='o', label='Progressive MSA', lw=2)
    ax2.fill_between(l_vals,
                     np.array(exp2['prog']['means']) - np.array(exp2['prog']['stds']),
                     np.array(exp2['prog']['means']) + np.array(exp2['prog']['stds']),
                     alpha=ALPHA, color=PROG_COLOR)
    ax2.plot(l_vals, exp2['star']['means'],
             color=STAR_COLOR, marker='s', label='Star Alignment', lw=2,
             linestyle='--')
    ax2.fill_between(l_vals,
                     np.array(exp2['star']['means']) - np.array(exp2['star']['stds']),
                     np.array(exp2['star']['means']) + np.array(exp2['star']['stds']),
                     alpha=ALPHA, color=STAR_COLOR)
    ax2.set_xlabel('Sequence Length (l)', fontsize=11)
    ax2.set_ylabel('Runtime (seconds)',   fontsize=11)
    ax2.set_title(f'Runtime vs l  (m={exp2["m"]})', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ── Plot 3: SP Score vs Mutation Rate ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mut_vals = exp3['mut_rates']
    ax3.plot(mut_vals, exp3['prog']['sp_means'],
             color=PROG_COLOR, marker='o', label='Progressive MSA', lw=2)
    ax3.fill_between(mut_vals,
                     np.array(exp3['prog']['sp_means']) - np.array(exp3['prog']['sp_stds']),
                     np.array(exp3['prog']['sp_means']) + np.array(exp3['prog']['sp_stds']),
                     alpha=ALPHA, color=PROG_COLOR)
    ax3.plot(mut_vals, exp3['star']['sp_means'],
             color=STAR_COLOR, marker='s', label='Star Alignment', lw=2,
             linestyle='--')
    ax3.fill_between(mut_vals,
                     np.array(exp3['star']['sp_means']) - np.array(exp3['star']['sp_stds']),
                     np.array(exp3['star']['sp_means']) + np.array(exp3['star']['sp_stds']),
                     alpha=ALPHA, color=STAR_COLOR)
    ax3.set_xlabel('Mutation Rate',       fontsize=11)
    ax3.set_ylabel('Sum-of-Pairs Score',  fontsize=11)
    ax3.set_title(f'Alignment Quality vs Mutation Rate\n'
                  f'(m={exp3["m"]}, l={exp3["l"]})', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # ── Plot 4: Log-log Runtime vs m (asymptotic validation) ─────────────
    ax4 = fig.add_subplot(gs[1, 1])
    m_arr   = np.array(m_vals, dtype=float)
    p_means = np.array(exp1['prog']['means'])
    s_means = np.array(exp1['star']['means'])

    # Only plot where runtimes are positive (avoid log(0))
    mask_p = p_means > 0
    mask_s = s_means > 0

    ax4.loglog(m_arr[mask_p], p_means[mask_p],
               color=PROG_COLOR, marker='o', label='Progressive MSA', lw=2)
    ax4.loglog(m_arr[mask_s], s_means[mask_s],
               color=STAR_COLOR, marker='s', label='Star Alignment', lw=2,
               linestyle='--')

    # Fit and plot O(m^2) reference line using progressive data
    if mask_p.sum() >= 2:
        log_m = np.log(m_arr[mask_p])
        log_t = np.log(p_means[mask_p])
        coeffs = np.polyfit(log_m, log_t, 1)
        slope  = coeffs[0]
        fit_t  = np.exp(np.polyval(coeffs, log_m))
        ax4.loglog(m_arr[mask_p], fit_t,
                   color='gray', linestyle=':', lw=1.5,
                   label=f'Fit: O(m^{slope:.2f})')

    ax4.set_xlabel('Number of Sequences m (log scale)', fontsize=11)
    ax4.set_ylabel('Runtime (log scale)',                fontsize=11)
    ax4.set_title('Log-Log: Asymptotic Validation',      fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, which='both', linestyle='--', alpha=0.5)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, 'msa_experiments.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plots saved to: {out_path}")
    plt.close()
    return out_path


# ─────────────────────────────────────────────
# 6.  Run All Experiments
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("MSA EXPERIMENTS — Starting benchmark suite")
    print("=" * 55)
    print("(This may take a few minutes for larger inputs...)\n")

    # ── Experiment 1: Runtime vs m ────────────────────────────────────────
    exp1 = experiment_runtime_vs_m(
        m_values = [2, 4, 6, 8, 10, 12, 14],
        l        = 50,
        mut_rate = 0.1,
        n_runs   = 3,
        seed     = 42
    )

    # ── Experiment 2: Runtime vs l ────────────────────────────────────────
    exp2 = experiment_runtime_vs_l(
        l_values = [20, 40, 60, 80, 100, 120, 150],
        m        = 6,
        mut_rate = 0.1,
        n_runs   = 3,
        seed     = 42
    )

    # ── Experiment 3: SP Score vs Mutation Rate ───────────────────────────
    exp3 = experiment_quality_vs_mutation(
        mut_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        m         = 8,
        l         = 60,
        n_runs    = 3,
        seed      = 42
    )

    # ── Generate all plots ────────────────────────────────────────────────
    print("\nGenerating plots ...")
    out_path = plot_all_results(exp1, exp2, exp3)

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY — Runtime vs m")
    print(f"  {'m':>4}  {'Prog (s)':>10}  {'Star (s)':>10}  {'Prog/Star':>10}")
    print("  " + "-" * 40)
    for i, m in enumerate(exp1['m_values']):
        p = exp1['prog']['means'][i]
        s = exp1['star']['means'][i]
        ratio = p / s if s > 0 else float('inf')
        print(f"  {m:>4}  {p:>10.4f}  {s:>10.4f}  {ratio:>10.2f}x")

    print("\nSUMMARY — SP Score vs Mutation Rate")
    print(f"  {'Rate':>6}  {'Prog SP':>10}  {'Star SP':>10}  {'Diff':>8}")
    print("  " + "-" * 40)
    for i, r in enumerate(exp3['mut_rates']):
        p = exp3['prog']['sp_means'][i]
        s = exp3['star']['sp_means'][i]
        print(f"  {r:>6.2f}  {p:>10.2f}  {s:>10.2f}  {p-s:>8.2f}")

    print("\n" + "=" * 55)
    print("All experiments complete!")
    print(f"Plots saved to: {out_path}")