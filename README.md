# Efficient Multiple Sequence Alignment in Bioinformatics

**A Progressive Heuristic Approach Using UPGMA Guide Trees and Profile-Profile Dynamic Programming**

> Final Project — Algorithms Design and Analysis (2026)
> Win Htut Naing (st126687) · Adviser: Dr. Chantri Polprasert · Asian Institute of Technology

---

## Overview

This project implements a complete **Multiple Sequence Alignment (MSA)** system from scratch in Python. It proposes and evaluates a progressive alignment algorithm against a Star Alignment baseline across four controlled experiments, confirming the theoretical O(m²l²) complexity and demonstrating a +6.6% alignment quality advantage at intermediate sequence divergence.

The system is organised into seven independent, unit-tested modules and supports three execution modes: a synthetic data demo, a full benchmarking suite, and real biological FASTA input. Three Tier 1 system improvements were implemented: ANSI colour-coded terminal output, FASTA output saving, and a detailed alignment statistics module with conserved region detection.

---

## Key Results

| Metric | Result |
|---|---|
| Complexity validation | Fitted exponent ≈ 2.1 (m-dim) and 2.0 (l-dim) — confirms O(m²l²) |
| Runtime advantage | 1.25× faster than Star Alignment at m = 14 |
| Quality advantage | +18.0 SP score (+6.6%) at mutation rate 0.25 |
| Alignment compactness | 32.3% fewer gap columns on real NCBI data |
| Unit tests | 29 / 29 passing across all 7 modules |

---

## Project Structure

```
MSA_Project/
├── main.py                        # Entry point — demo, FASTA, experiments modes
├── README.md
├── .gitignore
│
├── src/
│   ├── needleman_wunsch.py        # Module 1: Pairwise alignment (3-matrix affine gap DP)
│   ├── distance_matrix.py         # Module 2: Normalised pairwise distance matrix
│   ├── upgma.py                   # Module 3: UPGMA guide tree (min-heap clustering)
│   ├── profile.py                 # Module 4: Frequency matrix profiles + profile-profile DP
│   ├── progressive_msa.py         # Module 5: Full pipeline integration
│   ├── star_alignment.py          # Module 6: Star Alignment baseline
│   ├── data_generator.py          # Module 7a: Synthetic DNA sequence generation
│   ├── experiments.py             # Module 7b: Benchmarking suite
│   └── output_formatter.py        # Tier 1: ANSI terminal output + alignment statistics
│
├── data/
│   └── sequence.fasta             # Sample FASTA input (NCBI Cdx1 mRNA sequences)
│
└── results/
    └── msa_experiments.png        # Generated benchmark plot (4 panels)
```

---

## Requirements

Python 3.12+ is required. Install all dependencies with:

```bash
pip install numpy scipy matplotlib colorama
```

No external bioinformatics libraries are used. Every algorithmic component — from Needleman-Wunsch pairwise alignment to UPGMA clustering and profile-profile DP — is implemented directly from its mathematical definition.

---

## Usage

### Demo mode — synthetic sequences

```bash
python main.py
```

Generates 6 synthetic DNA sequences of base length 40 with mutation rate 0.15, runs both Progressive MSA and Star Alignment, and prints colour-coded alignment results with a side-by-side comparison panel.

### Demo mode with detailed statistics

```bash
python main.py --stats
```

Adds a detailed alignment statistics panel after each result, showing:
- Conserved / variable / gap column counts with bar charts
- Residue frequency breakdown (A, C, G, T, gap)
- Column-type minimap across the full alignment
- Conserved region detection ranked by biological significance

### FASTA mode — real biological sequences

```bash
python main.py --fasta data/sequence.fasta
```

Loads sequences from the specified FASTA file, runs both algorithms, prints colour-coded alignment results and a comparison panel, and automatically saves the progressive alignment to `data/sequence_aligned.fasta`.

### FASTA mode with custom output path

```bash
python main.py --fasta data/sequence.fasta --output results/my_alignment.fasta
```

### FASTA mode with statistics

```bash
python main.py --fasta data/sequence.fasta --stats
```

### Experiment mode — full benchmark suite

```bash
python main.py --experiments
```

Runs all four benchmark experiments and saves a four-panel plot to `results/msa_experiments.png`:
- **Panel (a):** Runtime vs number of sequences m (l = 50)
- **Panel (b):** Runtime vs sequence length l (m = 6)
- **Panel (c):** SP Score vs mutation rate (m = 8, l = 60)
- **Panel (d):** Log-log asymptotic fit (slope ≈ 2.1, confirming O(m²l²))

---

## Algorithm Pipeline

```
INPUT: m DNA sequences
        │
        ▼
┌──────────────────────┐
│ Needleman-Wunsch     │  ← All m(m-1)/2 pairwise alignments, O(l²) each
│ Pairwise Alignment   │
└─────────┬────────────┘
          │  pairwise scores
          ▼
┌──────────────────────┐
│ Distance Matrix      │  ← D(i,j) = 1 − score(i,j) / min(self_i, self_j)
└─────────┬────────────┘
          │  m×m distances in [0,1]
          ▼
┌──────────────────────┐
│ UPGMA Guide Tree     │  ← Min-heap clustering, O(m² log m)
└─────────┬────────────┘
          │  merge order (post-order traversal)
          ▼
┌──────────────────────┐
│ Profile-Profile      │  ← score(col_p, col_q) = Σₐ Σ_b freq_p[a]·freq_q[b]·sub(a,b)
│ Progressive Align    │
└─────────┬────────────┘
          │
          ▼
OUTPUT: Multiple Sequence Alignment + SP Score
```

**Complexity:** Time O(m²l²)  ·  Space O(m² + l²)

---

## Scoring Parameters

| Parameter | Value |
|---|---|
| Match score | +1.0 |
| Mismatch score | −1.0 |
| Gap open penalty | −2.0 |
| Gap extend penalty | −0.5 |
| Quality metric | Sum-of-Pairs (SP) score |

---

## Module Descriptions

| Module | Role | Tests |
|---|---|---|
| `needleman_wunsch.py` | Global pairwise alignment with three-matrix affine gap DP. Exposes `needleman_wunsch()` (full traceback) and `nw_score_only()` (score only, used in distance matrix phase). | 4 / 4 |
| `distance_matrix.py` | Builds the m×m symmetric distance matrix. Upper triangle only — O(m(m−1)/2) NW calls. | 4 / 4 |
| `upgma.py` | UPGMA hierarchical clustering using a `heapq` min-heap. Produces a binary `TreeNode` tree and `get_merge_order()` for post-order traversal. | 4 / 4 |
| `profile.py` | `Profile` class with per-residue frequency columns. `align_profiles()` implements profile-profile DP with the expected column score formula. `sum_of_pairs_score()` computes the SP score of any alignment. | 6 / 6 |
| `progressive_msa.py` | Integrates all four pipeline stages. `progressive_msa()` returns `{aligned_sequences, sp_score, runtime}`. `validate_alignment()` checks output correctness. | 4 / 4 |
| `star_alignment.py` | Baseline: center selection by maximum total score, then gap reconciliation by column-wise maximum gap count. | 3 / 3 |
| `data_generator.py` | `generate_sequences(m, l, mutation_rate, indel_rate, seed)` — applies substitutions, insertions, and deletions to a base sequence. Fixed seed for reproducibility. | 4 / 4 |
| `experiments.py` | `experiment_runtime_vs_m()`, `experiment_runtime_vs_l()`, `experiment_quality_vs_mutation()`, `plot_all_results()`. | — |
| `output_formatter.py` | ANSI colour-coded terminal output. `print_alignment_result()`, `print_comparison()`, `print_stats_summary()`, `find_conserved_regions()`, and helpers. Auto-detects colour support. | — |

---

## Bugs Fixed During Development

Three bugs were identified through unit testing and corrected before any experimental results were collected.

**Bug 1 — profile.py: traceback index out of bounds.** The traceback loop in `align_profiles()` did not check bounds before accessing sequence positions when aligning profiles of unequal length. Fixed by adding `if ia < len(seq)` guards at every sequence access in the traceback.

**Bug 2 — profile.py: unequal sequence lengths after merge.** The merged sequence list passed to `build_from_sequences()` could contain sequences of different lengths. Fixed by padding all sequences to the maximum length with gap characters before reconstruction.

**Bug 3 — star_alignment.py: index overflow at high mutation rates.** The gap reconciliation loop exceeded the valid range of the extended center sequence at mutation rates above 0.30. Fixed by clamping the loop bound to `min(len(gaps), center_len + 1)`.

---

## Tier 1 System Improvements

Three usability improvements were implemented after the core algorithm:

**1. ANSI colour-coded terminal output** (`output_formatter.py`)
Conserved columns → green · Mutation columns → yellow · Gap characters → red · Box-drawing panels for all results. Auto-falls back to plain text on non-TTY terminals (`NO_COLOR` env var supported).

**2. FASTA output saving** (`main.py --output`)
The progressive alignment is automatically saved as a valid FASTA file after every FASTA mode run. Use `--output FILE` to specify a custom path.

**3. Alignment statistics summary** (`main.py --stats`)
Detailed column-level analysis including bar charts, residue frequency breakdown, a column-type minimap, and a conserved region detector (`find_conserved_regions()`) that identifies biologically significant consecutive conserved positions ranked by length.

---

## Limitations

- **Python scalability:** Pure Python DP limits practical use to m ≤ 20 sequences and l ≤ 200 bp within ~5 seconds. NumPy vectorisation of the NW inner loop is the primary recommended improvement.
- **DNA only:** Simple match/mismatch matrix. Protein sequences require BLOSUM62 and extended residue alphabet.
- **UPGMA molecular clock:** Assumes equal evolutionary rates across lineages. Neighbour-Joining would be more appropriate for heterogeneous real biological data.
- **Internal quality metric only:** SP score is model-dependent. BAliBASE reference alignment evaluation is listed as future work.

---

## Future Work

1. NumPy vectorisation of the Needleman-Wunsch DP inner loop (est. 100–500× speedup)
2. Banded dynamic programming — reduce O(l²) to O(lk)
3. Neighbour-Joining guide tree — remove molecular clock assumption
4. Protein sequence support with BLOSUM62
5. Iterative refinement (MUSCLE-style) — re-estimate tree from aligned profiles
6. BAliBASE benchmark evaluation — model-independent quality metric

---

## References

- Needleman, S.B. and Wunsch, C.D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. *Journal of Molecular Biology*, 48(3), 443–453.
- Feng, D.F. and Doolittle, R.F. (1987). Progressive sequence alignment as a prerequisite to correct phylogenetic trees. *Journal of Molecular Evolution*, 25(4), 351–360.
- Thompson, J.D., Higgins, D.G. and Gibson, T.J. (1994). CLUSTAL W: improving the sensitivity of progressive multiple sequence alignment. *Nucleic Acids Research*, 22(22), 4673–4680.
- Notredame, C., Higgins, D.G. and Heringa, J. (2000). T-Coffee: A novel method for fast and accurate multiple sequence alignment. *Journal of Molecular Biology*, 302(1), 205–217.
- Edgar, R.C. (2004). MUSCLE: multiple sequence alignment with high accuracy and high throughput. *Nucleic Acids Research*, 32(5), 1792–1797.
- Wang, L. and Jiang, T. (1994). On the complexity of multiple sequence alignment. *Journal of Computational Biology*, 1(4), 337–348.

---

## Author

**Win Htut Naing** (st126687)
Algorithms Design and Analysis — AIT 2026
Adviser: Dr. Chantri Polprasert