# Efficient Multiple Sequence Alignment in Bioinformatics

**Course:** Algorithms Design and Analysis (ADA) — AIT 2026  
**Author:** Win Htut Naing (st126687)  
**Lecturer:** Dr. Chantri Polprasert  

## Overview

This project implements and compares two heuristic algorithms for the 
Multiple Sequence Alignment (MSA) problem:

- **Progressive MSA** (proposed) — uses a UPGMA guide tree and 
  profile-profile dynamic programming
- **Star Alignment** (baseline) — aligns all sequences to a central sequence

## Project Structure
```
MSA_Project/
├── main.py                  # Main entry point
├── src/
│   ├── needleman_wunsch.py  # Pairwise alignment with affine gap penalties
│   ├── distance_matrix.py   # Pairwise distance matrix computation
│   ├── upgma.py             # UPGMA guide tree construction
│   ├── profile.py           # Profile representation and alignment
│   ├── progressive_msa.py   # Full Progressive MSA pipeline
│   ├── star_alignment.py    # Star Alignment baseline
│   ├── data_generator.py    # Synthetic DNA sequence generator
│   └── experiments.py       # Benchmarking and plotting
├── data/                    # FASTA input files
└── results/                 # Experiment output plots
```

## Requirements
```bash
pip install numpy matplotlib biopython
```

## Usage
```bash
# Demo mode — runs on synthetic sequences
python main.py

# Run full benchmark experiments and generate plots
python main.py --experiments

# Align your own sequences from a FASTA file
python main.py --fasta data/test_cdx1.fasta
```

## Results Summary

| Metric | Progressive MSA | Star Alignment |
|--------|----------------|----------------|
| Runtime at m=14, l=50 | 1.577s | 1.970s |
| SP Score at mutation rate 0.25 | 291.0 | 273.0 |
| Alignment length (demo, m=6) | 42 cols | 44 cols |

## Algorithm Complexity

| Stage | Time Complexity | Space Complexity |
|-------|----------------|-----------------|
| Pairwise alignment | O(m²l²) | O(l²) |
| UPGMA tree | O(m² log m) | O(m²) |
| Profile merging | O(ml²) | O(mL) |
| **Total** | **O(m²l²)** | **O(m² + l²)** |