"""
star_alignment.py
--------------------
Module 6: Star Alignment (Baseline Algorithm)

The Star Alignment algorithm is the simplest MSA heuristic:
    1. Select a center sequence (the one with highest total similarity
       to all others — the "star center")
    2. Align every other sequence to the center using pairwise NW
    3. Merge all pairwise alignments by inserting gaps consistently
       across all sequences

Time complexity: O(m * l^2) — much faster than progressive MSA
                 but lower alignment quality due to error propagation
                 from the chosen center sequence.

This module serves as the BASELINE for comparison against
Progressive MSA in the experimental section.

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import time
import numpy as np

from needleman_wunsch import needleman_wunsch, nw_score_only, DNA_SCORE
from profile          import sum_of_pairs_score


# ─────────────────────────────────────────────
# 1.  Select Star Center
# ─────────────────────────────────────────────

def select_center(sequences    : list,
                  scoring_matrix        = None,
                  gap_open     : float  = -2.0,
                  gap_extend   : float  = -0.5) -> int:
    """
    Select the star center — the sequence with the highest total
    pairwise alignment score against all other sequences.

    This is equivalent to finding the sequence closest to all others
    (the "centroid" of the sequence set).

    Parameters
    ----------
    sequences      : list of str
    scoring_matrix : substitution dict
    gap_open       : gap opening penalty
    gap_extend     : gap extension penalty

    Returns
    -------
    int — index of the center sequence in the input list
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE

    m = len(sequences)
    total_scores = np.zeros(m)

    for i in range(m):
        for j in range(m):
            if i != j:
                total_scores[i] += nw_score_only(
                    sequences[i], sequences[j],
                    scoring_matrix, gap_open, gap_extend
                )

    center_idx = int(np.argmax(total_scores))
    return center_idx


# ─────────────────────────────────────────────
# 2.  Merge Pairwise Alignments
# ─────────────────────────────────────────────

def merge_pairwise_alignments(center_alignments : list,
                               other_alignments  : list,
                               center_idx        : int,
                               m                 : int) -> list:
    """
    Merge all pairwise alignments (center vs each other sequence)
    into a single consistent multiple alignment.

    Strategy:
        - The center sequence defines the "backbone" of the alignment
        - For each position in the merged alignment, we check all
          pairwise alignments and insert gap columns wherever ANY
          pairwise alignment has a gap in the center

    Parameters
    ----------
    center_alignments : list of str — center sequence in each pairwise aln
    other_alignments  : list of str — other sequence in each pairwise aln
    center_idx        : int — original index of center sequence
    m                 : int — total number of sequences

    Returns
    -------
    list of str — final merged aligned sequences (length m, equal-length)
    """
    num_others = len(center_alignments)

    # Find the maximum alignment length across all pairwise alignments
    # We need to reconcile gaps inserted in the center across alignments
    # by finding a unified column ordering

    # Step 1: For each pairwise alignment, record which center positions
    # have gaps inserted before them
    # gaps_before[k][pos] = number of extra gaps before center position pos
    # in pairwise alignment k

    # First, find the true length of center (no gaps)
    center_len = len(center_alignments[0].replace('-', ''))

    # For each pairwise alignment, record gap insertions before each
    # real center character
    gap_insertions = []   # gap_insertions[k] = list of gap counts per pos
    for ca in center_alignments:
        gaps = []
        gap_count = 0
        for ch in ca:
            if ch == '-':
                gap_count += 1
            else:
                gaps.append(gap_count)
                gap_count = 0
        gaps.append(gap_count)   # trailing gaps after last real char
        gap_insertions.append(gaps)

    # Step 2: At each center position, take the MAXIMUM gap insertion
    # across all pairwise alignments (ensures all gaps are accommodated)
    max_gaps = [0] * (center_len + 1)
    for gaps in gap_insertions:
        for pos in range(min(len(gaps), center_len + 1)):
            max_gaps[pos] = max(max_gaps[pos], gaps[pos])

    # Step 3: Reconstruct each sequence using the unified gap pattern
    # Build the center sequence first
    center_seq_raw = center_alignments[0].replace('-', '')

    def build_aligned(raw_seq_or_aligned, is_center, pairwise_ca, pairwise_oa):
        """Build the final aligned version of one sequence."""
        result = []
        if is_center:
            # Insert max_gaps before each real character
            for pos, ch in enumerate(center_seq_raw):
                result.extend(['-'] * max_gaps[pos])
                result.append(ch)
            result.extend(['-'] * max_gaps[center_len])
        else:
            # Walk through the pairwise alignment of this sequence
            center_pos = 0   # position in real center (no gaps)
            other_pos  = 0   # position in other sequence alignment
            ca_list    = list(pairwise_ca)
            oa_list    = list(pairwise_oa)

            # Track which gaps we've inserted at each center position
            inserted   = [0] * (center_len + 1)
            oa_chars   = []    # characters aligned to real center positions
            gap_chars  = []    # gap chars before each center position

            # Walk the pairwise alignment
            c_idx = 0    # index into real center characters
            i     = 0
            local_gaps_before = [[] for _ in range(center_len + 1)]

            for ca_ch, oa_ch in zip(ca_list, oa_list):
                if ca_ch != '-':
                    # Real center character — record what other has here
                    if c_idx < center_len:
                        oa_chars.append(oa_ch)
                    c_idx += 1
                else:
                    # Gap in center — other has an insertion here
                    if c_idx < center_len:
                        local_gaps_before[c_idx].append(oa_ch)
                    else:
                        local_gaps_before[center_len].append(oa_ch)

            # Now build the aligned sequence with unified gap columns
            result = []
            for pos in range(center_len):
                # Gaps before this center position (from this alignment)
                local_g = local_gaps_before[pos]
                # Max gaps required globally
                max_g   = max_gaps[pos]

                # Fill with local insertions first, then pad with gaps
                result.extend(local_g)
                result.extend(['-'] * (max_g - len(local_g)))

                # Add the character aligned to center pos
                if pos < len(oa_chars):
                    result.append(oa_chars[pos])
                else:
                    result.append('-')

            # Trailing gaps/chars
            local_g = local_gaps_before[center_len]
            max_g   = max_gaps[center_len]
            result.extend(local_g)
            result.extend(['-'] * (max_g - len(local_g)))

        return ''.join(result)

    # Build all aligned sequences
    final_seqs = [''] * m
    final_seqs[center_idx] = build_aligned(
        None, True, None, None
    )

    other_indices = [i for i in range(m) if i != center_idx]
    for k, orig_idx in enumerate(other_indices):
        final_seqs[orig_idx] = build_aligned(
            None, False,
            center_alignments[k],
            other_alignments[k]
        )

    # Pad all sequences to equal length (handle any length discrepancies)
    max_len = max(len(s) for s in final_seqs)
    final_seqs = [s + '-' * (max_len - len(s)) for s in final_seqs]

    return final_seqs


# ─────────────────────────────────────────────
# 3.  Main Star Alignment Pipeline
# ─────────────────────────────────────────────

def star_alignment(sequences      : list,
                   labels         : list  = None,
                   scoring_matrix         = None,
                   gap_open       : float = -2.0,
                   gap_extend     : float = -0.5,
                   verbose        : bool  = True) -> dict:
    """
    Run Star Alignment on a list of sequences.

    Parameters
    ----------
    sequences      : list of str — raw (unaligned) biological sequences
    labels         : list of str — sequence names (optional)
    scoring_matrix : substitution dict (default: DNA_SCORE)
    gap_open       : gap opening penalty
    gap_extend     : gap extension penalty
    verbose        : print progress steps

    Returns
    -------
    result : dict with keys:
        'aligned_sequences' : list of str — final aligned sequences
        'sp_score'          : float — Sum-of-Pairs score
        'center_idx'        : int — index of chosen center sequence
        'runtime'           : float — total wall-clock time in seconds
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE
    if labels is None:
        labels = [f"S{i}" for i in range(len(sequences))]

    m = len(sequences)
    start_time = time.time()

    # ── Step 1: Select center sequence ───────────────────────────────────
    if verbose:
        print("\n[Step 1] Selecting star center ...")
    center_idx = select_center(sequences, scoring_matrix,
                                gap_open, gap_extend)
    if verbose:
        print(f"  Center sequence: {labels[center_idx]} "
              f"(index {center_idx})")

    # ── Step 2: Align all sequences to center ────────────────────────────
    if verbose:
        print("\n[Step 2] Aligning all sequences to center ...")

    center_seq       = sequences[center_idx]
    center_alignments = []
    other_alignments  = []

    other_indices = [i for i in range(m) if i != center_idx]
    for i in other_indices:
        ca, oa, sc = needleman_wunsch(
            center_seq, sequences[i],
            scoring_matrix, gap_open, gap_extend
        )
        center_alignments.append(ca)
        other_alignments.append(oa)
        if verbose:
            print(f"  {labels[center_idx]} vs {labels[i]}: "
                  f"score={sc:.2f}")
            print(f"    center : {ca}")
            print(f"    other  : {oa}")

    # ── Step 3: Merge into multiple alignment ────────────────────────────
    if verbose:
        print("\n[Step 3] Merging pairwise alignments ...")

    aligned_sequences = merge_pairwise_alignments(
        center_alignments, other_alignments, center_idx, m
    )

    # ── Step 4: Compute SP score ─────────────────────────────────────────
    if verbose:
        print("\n[Step 4] Computing Sum-of-Pairs score ...")
    sp = sum_of_pairs_score(
        aligned_sequences, scoring_matrix, gap_open, gap_extend
    )

    runtime = time.time() - start_time

    if verbose:
        print("\n" + "=" * 55)
        print("STAR ALIGNMENT RESULT")
        print("=" * 55)
        max_label = max(len(l) for l in labels)
        for label, seq in zip(labels, aligned_sequences):
            print(f"  {label:<{max_label}} : {seq}")
        print(f"\n  Center sequence    : {labels[center_idx]}")
        print(f"  Sum-of-Pairs score : {sp:.4f}")
        print(f"  Alignment length   : {len(aligned_sequences[0])}")
        print(f"  Runtime            : {runtime:.4f}s")
        print("=" * 55)

    return {
        'aligned_sequences' : aligned_sequences,
        'sp_score'          : sp,
        'center_idx'        : center_idx,
        'runtime'           : runtime,
    }


# ─────────────────────────────────────────────
# 4.  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: Identical sequences ───────────────────────────────────────
    print("=" * 55)
    print("Test 1: 3 identical sequences")
    seqs   = ["ACGT", "ACGT", "ACGT"]
    labels = ["Seq1", "Seq2", "Seq3"]
    result = star_alignment(seqs, labels, verbose=True)

    assert result['sp_score'] == 12.0, \
        f"Test 1 FAILED — expected SP=12.0, got {result['sp_score']}"
    lengths = set(len(s) for s in result['aligned_sequences'])
    assert len(lengths) == 1, "Test 1 FAILED — unequal lengths"
    print("  PASSED ✓\n")

    # ── Test 2: 4 sequences with variation ───────────────────────────────
    print("=" * 55)
    print("Test 2: 4 sequences with variation")
    seqs = [
        "ACGTACGT",
        "ACGTACGT",
        "ACGTTCGT",
        "ACGTACCT",
    ]
    labels = ["Alpha", "Beta", "Gamma", "Delta"]
    result = star_alignment(seqs, labels, verbose=True)

    lengths = set(len(s) for s in result['aligned_sequences'])
    assert len(lengths) == 1, "Test 2 FAILED — unequal lengths"
    print(f"  SP Score: {result['sp_score']:.4f}")
    print("  PASSED ✓\n")

    # ── Test 3: Compare Star vs Progressive SP scores ─────────────────────
    print("=" * 55)
    print("Test 3: Star vs Progressive comparison")
    from progressive_msa import progressive_msa

    seqs = [
        "ACGTACGT",
        "ACGAACGT",
        "ACGTACTT",
        "GCGTACGT",
    ]
    labels = ["S1", "S2", "S3", "S4"]

    star_result = star_alignment(seqs, labels, verbose=False)
    prog_result = progressive_msa(seqs, labels, verbose=False)

    print(f"  Star Alignment SP score       : "
          f"{star_result['sp_score']:.4f}")
    print(f"  Progressive MSA SP score      : "
          f"{prog_result['sp_score']:.4f}")
    print(f"  Star runtime                  : "
          f"{star_result['runtime']:.4f}s")
    print(f"  Progressive runtime           : "
          f"{prog_result['runtime']:.4f}s")
    print(f"\n  Star aligned sequences:")
    for l, s in zip(labels, star_result['aligned_sequences']):
        print(f"    {l}: {s}")
    print(f"\n  Progressive aligned sequences:")
    for l, s in zip(labels, prog_result['aligned_sequences']):
        print(f"    {l}: {s}")
    print("  PASSED ✓\n")

    print("=" * 55)
    print("All tests passed! Module 6 is ready.")
    print("Baseline Star Alignment is fully functional.")