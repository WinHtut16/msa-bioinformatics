"""
profile.py
--------------------
Module 4: Profile Representation and Profile-Profile Alignment

A Profile is a 2D frequency matrix representing a group of aligned sequences.
Each column of the profile stores the frequency of each character at that position.

Example — 3 aligned sequences:
    ACGT
    A-GT
    ACGT
Profile column 0: {A:1.0, C:0.0, G:0.0, T:0.0, -:0.0}
Profile column 1: {A:0.0, C:0.67, G:0.0, T:0.0, -:0.33}
...

Profile-Profile alignment uses DP where the substitution score between
two profile columns is the expected substitution score:
    score(col_p, col_q) = sum over (a,b): freq_p[a] * freq_q[b] * sub(a,b)

Gap columns in profiles are handled using a gap frequency penalty.

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import numpy as np
from needleman_wunsch import DNA_SCORE


# ─────────────────────────────────────────────
# 1.  Alphabet & Profile Structure
# ─────────────────────────────────────────────

DNA_ALPHABET = ['A', 'C', 'G', 'T', '-']
CHAR_TO_IDX  = {c: i for i, c in enumerate(DNA_ALPHABET)}


class Profile:
    """
    Represents a multiple sequence alignment as a frequency matrix.

    Attributes
    ----------
    freq_matrix : np.ndarray of shape (len(alphabet), L)
                  freq_matrix[char_idx][col] = frequency of char at column col
    length      : int — number of alignment columns (L)
    num_seqs    : int — number of sequences contributing to this profile
    alphabet    : list of str — characters including gap '-'
    sequences   : list of str — the actual aligned sequences (with gaps)
    """

    def __init__(self, alphabet: list = None):
        self.alphabet    = alphabet or DNA_ALPHABET
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.freq_matrix = None   # shape: (len(alphabet), L)
        self.length      = 0
        self.num_seqs    = 0
        self.sequences   = []     # aligned sequences stored here

    def build_from_sequences(self, sequences: list) -> None:
        """
        Build a profile frequency matrix from a list of aligned sequences.
        All sequences must have equal length (gaps already inserted).

        Parameters
        ----------
        sequences : list of str — equal-length aligned sequences
        """
        assert len(set(len(s) for s in sequences)) == 1, \
            "All sequences must have equal length to build a profile"

        self.sequences = list(sequences)
        self.num_seqs  = len(sequences)
        self.length    = len(sequences[0])
        n_chars        = len(self.alphabet)

        self.freq_matrix = np.zeros((n_chars, self.length), dtype=float)

        for seq in sequences:
            for col, char in enumerate(seq):
                if char in self.char_to_idx:
                    self.freq_matrix[self.char_to_idx[char]][col] += 1.0

        # Normalise columns to frequencies (sum to 1.0)
        col_sums = self.freq_matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1.0   # avoid division by zero
        self.freq_matrix /= col_sums

    def get_column(self, col: int) -> dict:
        """
        Return frequency dict for a given column index.

        Returns
        -------
        dict: {char -> frequency}
        """
        return {c: self.freq_matrix[i][col]
                for i, c in enumerate(self.alphabet)}

    def __repr__(self):
        return (f"Profile(length={self.length}, "
                f"num_seqs={self.num_seqs})")


# ─────────────────────────────────────────────
# 2.  Column-Column Scoring
# ─────────────────────────────────────────────

def score_columns(col_p: dict, col_q: dict,
                  scoring_matrix: dict = None) -> float:
    """
    Compute expected substitution score between two profile columns.

    score = sum over all (a, b) pairs:
                freq_p[a] * freq_q[b] * substitution_score(a, b)

    Gap characters '-' are excluded from substitution scoring
    (gaps are handled by gap penalties in the DP).

    Parameters
    ----------
    col_p, col_q    : dict {char -> frequency}
    scoring_matrix  : substitution scoring dict

    Returns
    -------
    float — expected score between the two columns
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE

    total = 0.0
    for a, fa in col_p.items():
        if a == '-' or fa == 0.0:
            continue
        for b, fb in col_q.items():
            if b == '-' or fb == 0.0:
                continue
            sub = scoring_matrix.get((a, b),
                  scoring_matrix.get((b, a), -1.0))
            total += fa * fb * sub

    return total


# ─────────────────────────────────────────────
# 3.  Profile-Profile Alignment (DP)
# ─────────────────────────────────────────────

def align_profiles(profile_a: Profile,
                   profile_b: Profile,
                   scoring_matrix: dict = None,
                   gap_open: float = -2.0,
                   gap_extend: float = -0.5) -> tuple:
    """
    Align two profiles using dynamic programming with affine gap penalties.

    The DP recurrence mirrors Needleman-Wunsch but operates on profile
    columns instead of individual characters.

    Parameters
    ----------
    profile_a, profile_b : Profile objects to align
    scoring_matrix       : substitution scoring dict
    gap_open             : gap opening penalty
    gap_extend           : gap extension penalty

    Returns
    -------
    (merged_profile, score) : tuple
        merged_profile : Profile — new profile combining both inputs
        score          : float  — total alignment score
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE

    La = profile_a.length
    Lb = profile_b.length
    NEG_INF = float('-inf')

    # ── Initialise DP matrices (same 3-matrix affine gap approach) ────────
    M = np.full((La + 1, Lb + 1), NEG_INF)
    X = np.full((La + 1, Lb + 1), NEG_INF)
    Y = np.full((La + 1, Lb + 1), NEG_INF)

    trace_M = np.zeros((La + 1, Lb + 1), dtype=int)
    trace_X = np.zeros((La + 1, Lb + 1), dtype=int)
    trace_Y = np.zeros((La + 1, Lb + 1), dtype=int)

    M[0][0] = 0.0
    for i in range(1, La + 1):
        X[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, Lb + 1):
        Y[0][j] = gap_open + (j - 1) * gap_extend

    # ── Fill DP ───────────────────────────────────────────────────────────
    for i in range(1, La + 1):
        col_a = profile_a.get_column(i - 1)
        for j in range(1, Lb + 1):
            col_b = profile_b.get_column(j - 1)

            sub = score_columns(col_a, col_b, scoring_matrix)

            # M: match/mismatch between column i and column j
            candidates_M = [M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]]
            best_M = max(candidates_M)
            M[i][j] = best_M + sub
            trace_M[i][j] = candidates_M.index(best_M)

            # X: gap in profile_b
            open_X   = M[i-1][j] + gap_open
            extend_X = X[i-1][j] + gap_extend
            if extend_X >= open_X:
                X[i][j] = extend_X; trace_X[i][j] = 1
            else:
                X[i][j] = open_X;   trace_X[i][j] = 0

            # Y: gap in profile_a
            open_Y   = M[i][j-1] + gap_open
            extend_Y = Y[i][j-1] + gap_extend
            if extend_Y >= open_Y:
                Y[i][j] = extend_Y; trace_Y[i][j] = 2
            else:
                Y[i][j] = open_Y;   trace_Y[i][j] = 0

    # ── Choose best terminal state ────────────────────────────────────────
    terminal   = [M[La][Lb], X[La][Lb], Y[La][Lb]]
    best_score = max(terminal)
    state      = terminal.index(best_score)

    # ── Traceback ─────────────────────────────────────────────────────────
    # Record alignment as operations: 'M'=match, 'X'=gap in B, 'Y'=gap in A
    ops = []
    i, j = La, Lb

    while i > 0 or j > 0:
        if state == 0:
            ops.append('M')
            state = int(trace_M[i][j])
            i -= 1; j -= 1
        elif state == 1:
            ops.append('X')
            state = int(trace_X[i][j])
            i -= 1
        else:
            ops.append('Y')
            state = int(trace_Y[i][j])
            j -= 1

    ops.reverse()

    # ── Build merged alignment ────────────────────────────────────────────
    gap_col = '-'
    merged_sequences = []

    # Initialise with copies of existing sequences
    seqs_a = [list(s) for s in profile_a.sequences]
    seqs_b = [list(s) for s in profile_b.sequences]

    aligned_a = [[] for _ in seqs_a]
    aligned_b = [[] for _ in seqs_b]

    ia, ib = 0, 0
    for op in ops:
        if op == 'M':
            # Take column ia from A, column ib from B
            for k, seq in enumerate(seqs_a):
                aligned_a[k].append(seq[ia] if ia < len(seq) else gap_col)
            for k, seq in enumerate(seqs_b):
                aligned_b[k].append(seq[ib] if ib < len(seq) else gap_col)
            ia += 1; ib += 1
        elif op == 'X':
            # Gap column inserted into B, take real column from A
            for k, seq in enumerate(seqs_a):
                aligned_a[k].append(seq[ia] if ia < len(seq) else gap_col)
            for k in range(len(seqs_b)):
                aligned_b[k].append(gap_col)
            ia += 1
        else:  # 'Y'
            # Gap column inserted into A, take real column from B
            for k in range(len(seqs_a)):
                aligned_a[k].append(gap_col)
            for k, seq in enumerate(seqs_b):
                aligned_b[k].append(seq[ib] if ib < len(seq) else gap_col)
            ib += 1

    # Convert back to strings
    final_seqs_a = [''.join(row) for row in aligned_a]
    final_seqs_b = [''.join(row) for row in aligned_b]
    merged_sequences = final_seqs_a + final_seqs_b

    # Ensure all merged sequences are equal length (pad with gaps)
    max_len = max(len(s) for s in merged_sequences)
    merged_sequences = [s + gap_col * (max_len - len(s)) for s in merged_sequences]

    # Build merged profile
    merged_profile = Profile(alphabet=profile_a.alphabet)
    merged_profile.build_from_sequences(merged_sequences)

    return merged_profile, best_score


# ─────────────────────────────────────────────
# 4.  Helper: sequence → single-seq profile
# ─────────────────────────────────────────────

def sequence_to_profile(seq: str, alphabet: list = None) -> Profile:
    """
    Wrap a single sequence in a Profile object.
    Used to initialise leaf nodes before progressive merging.
    """
    p = Profile(alphabet=alphabet or DNA_ALPHABET)
    p.build_from_sequences([seq])
    return p


# ─────────────────────────────────────────────
# 5.  Sum-of-Pairs Score
# ─────────────────────────────────────────────

def sum_of_pairs_score(sequences: list,
                       scoring_matrix: dict = None,
                       gap_open: float = -2.0,
                       gap_extend: float = -0.5) -> float:
    """
    Compute the Sum-of-Pairs (SP) score for a multiple sequence alignment.

    SP(A) = sum over all pairs (i,j): score(aligned_i, aligned_j)

    This is the standard quality metric for MSA evaluation.

    Parameters
    ----------
    sequences : list of str — equal-length aligned sequences (with gaps)

    Returns
    -------
    float — total SP score
    """
    from needleman_wunsch import needleman_wunsch
    total = 0.0
    m = len(sequences)
    for i in range(m):
        for j in range(i + 1, m):
            _, _, sc = needleman_wunsch(
                sequences[i].replace('-', ''),
                sequences[j].replace('-', ''),
                scoring_matrix, gap_open, gap_extend
            )
            total += sc
    return total


# ─────────────────────────────────────────────
# 6.  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: Profile from identical sequences ───────────────────────────
    print("=" * 55)
    print("Test 1: Profile from identical sequences")
    seqs = ["ACGT", "ACGT", "ACGT"]
    p = Profile()
    p.build_from_sequences(seqs)
    print(f"  Profile: {p}")
    col0 = p.get_column(0)
    print(f"  Column 0 frequencies: {col0}")
    assert col0['A'] == 1.0, "Test 1 FAILED — expected A=1.0 at col 0"
    assert col0['C'] == 0.0, "Test 1 FAILED — expected C=0.0 at col 0"
    print("  PASSED ✓\n")

    # ── Test 2: Profile with gaps ──────────────────────────────────────────
    print("=" * 55)
    print("Test 2: Profile with gaps")
    seqs = ["ACGT", "A-GT", "ACGT"]
    p = Profile()
    p.build_from_sequences(seqs)
    col1 = p.get_column(1)
    print(f"  Column 1 frequencies: {col1}")
    assert abs(col1['C'] - 2/3) < 1e-6, "Test 2 FAILED — C freq wrong"
    assert abs(col1['-'] - 1/3) < 1e-6, "Test 2 FAILED — gap freq wrong"
    print("  PASSED ✓\n")

    # ── Test 3: Column-column scoring ─────────────────────────────────────
    print("=" * 55)
    print("Test 3: Column-column scoring")
    col_a = {'A': 1.0, 'C': 0.0, 'G': 0.0, 'T': 0.0, '-': 0.0}
    col_b = {'A': 1.0, 'C': 0.0, 'G': 0.0, 'T': 0.0, '-': 0.0}
    sc = score_columns(col_a, col_b)
    print(f"  A vs A score: {sc}  (expected: 1.0)")
    assert sc == 1.0, "Test 3 FAILED"

    col_c = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 1.0, '-': 0.0}
    sc2 = score_columns(col_a, col_c)
    print(f"  A vs T score: {sc2}  (expected: -1.0)")
    assert sc2 == -1.0, "Test 3 FAILED"
    print("  PASSED ✓\n")

    # ── Test 4: Profile-profile alignment ─────────────────────────────────
    print("=" * 55)
    print("Test 4: Profile-profile alignment")
    p1 = sequence_to_profile("ACGT")
    p2 = sequence_to_profile("ACGT")
    merged, sc = align_profiles(p1, p2)
    print(f"  Merged profile  : {merged}")
    print(f"  Alignment score : {sc}  (expected: 4.0)")
    print(f"  Merged sequences: {merged.sequences}")
    assert sc == 4.0, "Test 4 FAILED — identical profiles should score 4.0"
    assert len(set(len(s) for s in merged.sequences)) == 1, \
        "Test 4 FAILED — merged sequences must have equal length"
    print("  PASSED ✓\n")

    # ── Test 5: Profile-profile alignment with gap ─────────────────────────
    print("=" * 55)
    print("Test 5: Profile-profile alignment with insertion")
    p1 = sequence_to_profile("ACGT")
    p2 = sequence_to_profile("AGT")
    merged, sc = align_profiles(p1, p2)
    print(f"  Merged sequences: {merged.sequences}")
    print(f"  Alignment score : {sc}")
    assert '-' in merged.sequences[1], \
        "Test 5 FAILED — expected gap inserted in shorter sequence"
    assert len(merged.sequences[0]) == len(merged.sequences[1]), \
        "Test 5 FAILED — sequences must be equal length after alignment"
    print("  PASSED ✓\n")

    # ── Test 6: SP score on perfect alignment ─────────────────────────────
    print("=" * 55)
    print("Test 6: Sum-of-Pairs score")
    aligned = ["ACGT", "ACGT", "ACGT"]
    sp = sum_of_pairs_score(aligned)
    print(f"  SP score (3 identical): {sp}  (expected: 12.0)")
    assert sp == 12.0, "Test 6 FAILED"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("All tests passed! Module 4 is ready.")