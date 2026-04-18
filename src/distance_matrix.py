"""
distance_matrix.py
--------------------
Module 2: Pairwise Distance Matrix Computation

Computes an m x m distance matrix from m biological sequences.

Steps:
    1. Run Needleman-Wunsch on every pair (i, j) to get alignment score
    2. Normalize the score into a distance value in range [0, 1]
       using the formula:
           dist(i,j) = 1 - (score(i,j) / min(self_score(i), self_score(j)))
       where self_score(x) = score of aligning x with itself (perfect score)

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import numpy as np
from needleman_wunsch import nw_score_only, DNA_SCORE


# ─────────────────────────────────────────────
# 1.  Self-score helper
# ─────────────────────────────────────────────

def self_score(seq: str,
               scoring_matrix: dict = None,
               gap_open: float = -2.0,
               gap_extend: float = -0.5) -> float:
    """
    Compute the score of aligning a sequence with itself (perfect alignment).
    This is used as a normalisation reference.

    For DNA with match=1, self_score("ACGT") = 4.0
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE
    return nw_score_only(seq, seq, scoring_matrix, gap_open, gap_extend)


# ─────────────────────────────────────────────
# 2.  Core: Build Distance Matrix
# ─────────────────────────────────────────────

def compute_distance_matrix(sequences: list,
                             scoring_matrix: dict = None,
                             gap_open: float = -2.0,
                             gap_extend: float = -0.5) -> np.ndarray:
    """
    Compute a symmetric m x m distance matrix for a list of sequences.

    Distance formula (kimura-style normalisation):
        dist(i, j) = 1 - (score(i,j) / min(self_score(i), self_score(j)))

    Values are clipped to [0, 1] to handle edge cases where score > self_score
    (can happen with very short or degenerate sequences).

    Parameters
    ----------
    sequences      : list of strings (biological sequences)
    scoring_matrix : substitution scoring dict
    gap_open       : gap opening penalty
    gap_extend     : gap extension penalty

    Returns
    -------
    dist_matrix : np.ndarray of shape (m, m), dtype float64
                  dist_matrix[i][j] = distance between seq i and seq j
                  Diagonal is 0.0 (a sequence is identical to itself)
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE

    m = len(sequences)
    dist_matrix = np.zeros((m, m), dtype=float)

    # Pre-compute all self-scores to avoid recomputation
    self_scores = [self_score(seq, scoring_matrix, gap_open, gap_extend)
                   for seq in sequences]

    print(f"  Computing {m*(m-1)//2} pairwise alignments for {m} sequences...")

    for i in range(m):
        for j in range(i + 1, m):
            # Pairwise alignment score
            pair_score = nw_score_only(sequences[i], sequences[j],
                                       scoring_matrix, gap_open, gap_extend)

            # Normalise: use the smaller self-score as reference
            ref = min(self_scores[i], self_scores[j])

            if ref <= 0:
                # Edge case: degenerate sequence, set max distance
                dist = 1.0
            else:
                dist = 1.0 - (pair_score / ref)
                dist = float(np.clip(dist, 0.0, 1.0))

            # Symmetric matrix
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix


# ─────────────────────────────────────────────
# 3.  Pretty-print helper
# ─────────────────────────────────────────────

def print_distance_matrix(dist_matrix: np.ndarray,
                           labels: list = None) -> None:
    """
    Print the distance matrix in a readable format.

    Parameters
    ----------
    dist_matrix : np.ndarray — the m x m distance matrix
    labels      : list of str — sequence names/labels (optional)
    """
    m = dist_matrix.shape[0]
    if labels is None:
        labels = [f"S{i+1}" for i in range(m)]

    col_width = 8
    header = " " * 4 + "".join(f"{l:>{col_width}}" for l in labels)
    print(header)
    print(" " * 4 + "-" * col_width * m)

    for i in range(m):
        row = f"{labels[i]:<4}"
        for j in range(m):
            row += f"{dist_matrix[i][j]:>{col_width}.4f}"
        print(row)


# ─────────────────────────────────────────────
# 4.  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("Test 1: Identical sequences → distance should be 0.0")
    seqs = ["ACGT", "ACGT", "ACGT"]
    dm = compute_distance_matrix(seqs)
    print_distance_matrix(dm, labels=["S1", "S2", "S3"])
    assert np.allclose(dm, 0.0), "Test 1 FAILED — expected all zeros"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("Test 2: Varying similarity")
    seqs = [
        "ACGTACGT",   # S1 — reference
        "ACGTACGT",   # S2 — identical to S1
        "ACGTTTTT",   # S3 — half similar
        "TTTTTTTT",   # S4 — very different
    ]
    labels = ["S1", "S2", "S3", "S4"]
    dm = compute_distance_matrix(seqs, gap_open=-2.0, gap_extend=-0.5)
    print_distance_matrix(dm, labels=labels)

    # S1 vs S2 should be 0
    assert dm[0][1] == 0.0, "Test 2 FAILED — identical seqs should have dist=0"
    # S1 vs S4 should be greater than S1 vs S3
    assert dm[0][3] > dm[0][2], \
        "Test 2 FAILED — S1-S4 dist should be > S1-S3 dist"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("Test 3: Single sequence edge case")
    seqs = ["ACGT"]
    dm = compute_distance_matrix(seqs)
    print_distance_matrix(dm, labels=["S1"])
    assert dm.shape == (1, 1) and dm[0][0] == 0.0, "Test 3 FAILED"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("Test 4: Matrix symmetry check")
    seqs = ["ACGT", "AGTT", "CCCC", "GGGG"]
    dm = compute_distance_matrix(seqs)
    assert np.allclose(dm, dm.T), "Test 4 FAILED — matrix is not symmetric"
    print("  Matrix is symmetric ✓")
    print("  Diagonal is all zeros ✓")
    print("  PASSED ✓\n")

    print("=" * 55)
    print("All tests passed! Module 2 is ready.")