"""
needleman_wunsch.py
--------------------
Module 1: Pairwise Global Sequence Alignment using Needleman-Wunsch
with Affine Gap Penalties.

Affine gap penalty model:
    gap_cost = gap_open + (gap_length * gap_extend)

Three DP matrices are used:
    M[i][j]  = best score when seq1[i] and seq2[j] are MATCHED/MISMATCHED
    X[i][j]  = best score when seq1[i] is aligned to a GAP (gap in seq2)
    Y[i][j]  = best score when seq2[j] is aligned to a GAP (gap in seq1)

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import numpy as np


# ─────────────────────────────────────────────
# 1.  Scoring Matrices
# ─────────────────────────────────────────────

# Simple DNA match/mismatch scoring
DNA_SCORE = {
    ('A', 'A'): 1,  ('A', 'C'): -1, ('A', 'G'): -1, ('A', 'T'): -1,
    ('C', 'A'): -1, ('C', 'C'): 1,  ('C', 'G'): -1, ('C', 'T'): -1,
    ('G', 'A'): -1, ('G', 'C'): -1, ('G', 'G'): 1,  ('G', 'T'): -1,
    ('T', 'A'): -1, ('T', 'C'): -1, ('T', 'G'): -1, ('T', 'T'): 1,
}


def get_substitution_score(a: str, b: str, scoring_matrix: dict) -> float:
    """
    Return the substitution score for aligning character 'a' with character 'b'.

    Parameters
    ----------
    a, b           : single characters to compare
    scoring_matrix : dict mapping (char, char) -> score

    Returns
    -------
    float score
    """
    return scoring_matrix.get((a, b), scoring_matrix.get((b, a), -1))


# ─────────────────────────────────────────────
# 2.  Core DP — Needleman-Wunsch (Affine Gaps)
# ─────────────────────────────────────────────

def needleman_wunsch(seq1: str,
                     seq2: str,
                     scoring_matrix: dict = None,
                     gap_open: float = -2.0,
                     gap_extend: float = -0.5) -> tuple:
    """
    Global pairwise alignment using Needleman-Wunsch with affine gap penalties.

    Parameters
    ----------
    seq1, seq2      : biological sequences (strings of characters)
    scoring_matrix  : substitution scoring dict, defaults to DNA_SCORE
    gap_open        : penalty for opening a new gap (negative float)
    gap_extend      : penalty for extending an existing gap (negative float)

    Returns
    -------
    (aligned_seq1, aligned_seq2, score) : tuple
        aligned_seq1 : str  — seq1 with inserted gaps '-'
        aligned_seq2 : str  — seq2 with inserted gaps '-'
        score        : float — total alignment score
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE

    n = len(seq1)
    m = len(seq2)
    NEG_INF = float('-inf')

    # ── Initialise three DP matrices ──────────────────────────────────────
    # M[i][j] : seq1[i-1] aligned to seq2[j-1]
    # X[i][j] : gap in seq2 (seq1[i-1] aligned to '-')
    # Y[i][j] : gap in seq1 ('-' aligned to seq2[j-1])
    M = np.full((n + 1, m + 1), NEG_INF)
    X = np.full((n + 1, m + 1), NEG_INF)
    Y = np.full((n + 1, m + 1), NEG_INF)

    # Traceback matrices: store which state we came from
    # 0=M, 1=X, 2=Y
    trace_M = np.zeros((n + 1, m + 1), dtype=int)
    trace_X = np.zeros((n + 1, m + 1), dtype=int)
    trace_Y = np.zeros((n + 1, m + 1), dtype=int)

    # ── Base cases ────────────────────────────────────────────────────────
    M[0][0] = 0.0
    for i in range(1, n + 1):
        X[i][0] = gap_open + (i - 1) * gap_extend   # open + extend*(len-1)
        M[i][0] = NEG_INF
        Y[i][0] = NEG_INF
    for j in range(1, m + 1):
        Y[0][j] = gap_open + (j - 1) * gap_extend
        M[0][j] = NEG_INF
        X[0][j] = NEG_INF

    # ── Fill DP tables ────────────────────────────────────────────────────
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = get_substitution_score(seq1[i - 1], seq2[j - 1],
                                         scoring_matrix)

            # --- M[i][j]: best alignment ending with a match/mismatch ----
            candidates_M = [M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]]
            best_M = max(candidates_M)
            M[i][j] = best_M + sub
            trace_M[i][j] = candidates_M.index(best_M)  # 0,1,2 → M,X,Y

            # --- X[i][j]: gap in seq2 (extending or opening in seq1) -----
            open_X   = M[i-1][j] + gap_open    # open new gap from M
            extend_X = X[i-1][j] + gap_extend  # extend existing gap
            if extend_X >= open_X:
                X[i][j] = extend_X
                trace_X[i][j] = 1              # came from X
            else:
                X[i][j] = open_X
                trace_X[i][j] = 0              # came from M

            # --- Y[i][j]: gap in seq1 (extending or opening in seq2) -----
            open_Y   = M[i][j-1] + gap_open
            extend_Y = Y[i][j-1] + gap_extend
            if extend_Y >= open_Y:
                Y[i][j] = extend_Y
                trace_Y[i][j] = 2              # came from Y
            else:
                Y[i][j] = open_Y
                trace_Y[i][j] = 0              # came from M

    # ── Choose best terminal state ────────────────────────────────────────
    terminal = [M[n][m], X[n][m], Y[n][m]]
    best_score = max(terminal)
    state = terminal.index(best_score)   # 0=M, 1=X, 2=Y

    # ── Traceback ─────────────────────────────────────────────────────────
    aligned1, aligned2 = [], []
    i, j = n, m

    while i > 0 or j > 0:
        if state == 0:          # came through M: match/mismatch
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            state = int(trace_M[i][j])
            i -= 1
            j -= 1

        elif state == 1:        # came through X: gap in seq2
            aligned1.append(seq1[i - 1])
            aligned2.append('-')
            state = int(trace_X[i][j])
            i -= 1

        else:                   # state == 2, came through Y: gap in seq1
            aligned1.append('-')
            aligned2.append(seq2[j - 1])
            state = int(trace_Y[i][j])
            j -= 1

    aligned_seq1 = ''.join(reversed(aligned1))
    aligned_seq2 = ''.join(reversed(aligned2))

    return aligned_seq1, aligned_seq2, best_score


# ─────────────────────────────────────────────
# 3.  Score-only version (no traceback)
# ─────────────────────────────────────────────

def nw_score_only(seq1: str,
                  seq2: str,
                  scoring_matrix: dict = None,
                  gap_open: float = -2.0,
                  gap_extend: float = -0.5) -> float:
    """
    Returns only the alignment score (no traceback).
    Slightly faster — used for building the distance matrix.
    """
    _, _, score = needleman_wunsch(seq1, seq2, scoring_matrix,
                                   gap_open, gap_extend)
    return score


# ─────────────────────────────────────────────
# 4.  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Test 1: Identical sequences (score should be length * match_score) ─
    s1, s2 = "ACGT", "ACGT"
    a1, a2, sc = needleman_wunsch(s1, s2)
    print("=" * 50)
    print("Test 1: Identical sequences")
    print(f"  seq1     : {s1}")
    print(f"  seq2     : {s2}")
    print(f"  aligned1 : {a1}")
    print(f"  aligned2 : {a2}")
    print(f"  score    : {sc}  (expected: 4.0)")
    assert a1 == "ACGT" and a2 == "ACGT" and sc == 4.0, "Test 1 FAILED"
    print("  PASSED ✓")

    # ── Test 2: One insertion ──────────────────────────────────────────────
    s1, s2 = "ACGT", "AGT"
    a1, a2, sc = needleman_wunsch(s1, s2)
    print("\nTest 2: One insertion/deletion")
    print(f"  seq1     : {s1}")
    print(f"  seq2     : {s2}")
    print(f"  aligned1 : {a1}")
    print(f"  aligned2 : {a2}")
    print(f"  score    : {sc}")
    assert len(a1) == len(a2), "Test 2 FAILED — lengths differ"
    assert '-' in a2, "Test 2 FAILED — expected gap in seq2"
    print("  PASSED ✓")

    # ── Test 3: Completely different sequences ─────────────────────────────
    s1, s2 = "AAAA", "TTTT"
    a1, a2, sc = needleman_wunsch(s1, s2)
    print("\nTest 3: Completely different sequences")
    print(f"  seq1     : {s1}")
    print(f"  seq2     : {s2}")
    print(f"  aligned1 : {a1}")
    print(f"  aligned2 : {a2}")
    print(f"  score    : {sc}  (expected: -4.0)")
    assert sc == -4.0, "Test 3 FAILED"
    print("  PASSED ✓")

    # ── Test 4: Gap extension test ─────────────────────────────────────────
    s1, s2 = "ACCCGT", "AGT"
    a1, a2, sc = needleman_wunsch(s1, s2)
    print("\nTest 4: Gap extension")
    print(f"  seq1     : {s1}")
    print(f"  seq2     : {s2}")
    print(f"  aligned1 : {a1}")
    print(f"  aligned2 : {a2}")
    print(f"  score    : {sc}")
    assert len(a1) == len(a2), "Test 4 FAILED — lengths differ"
    print("  PASSED ✓")

    print("\n" + "=" * 50)
    print("All tests passed! Module 1 is ready.")