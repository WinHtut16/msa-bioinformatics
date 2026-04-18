"""
progressive_msa.py
--------------------
Module 5: Full Progressive Multiple Sequence Alignment Pipeline

This module ties together all previous modules:
    1. needleman_wunsch.py  — pairwise alignment + scoring
    2. distance_matrix.py   — pairwise distance computation
    3. upgma.py             — guide tree construction
    4. profile.py           — profile representation + profile-profile alignment

Pipeline:
    sequences
        → distance matrix (Module 2)
        → UPGMA guide tree (Module 3)
        → bottom-up profile merging (Module 4)
        → final multiple sequence alignment

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import time
import numpy as np

from needleman_wunsch import DNA_SCORE
from distance_matrix  import compute_distance_matrix
from upgma            import build_upgma_tree, get_merge_order, print_tree
from profile          import (Profile, sequence_to_profile,
                               align_profiles, sum_of_pairs_score)


# ─────────────────────────────────────────────
# 1.  Main Pipeline
# ─────────────────────────────────────────────

def progressive_msa(sequences : list,
                    labels     : list  = None,
                    scoring_matrix     = None,
                    gap_open   : float = -2.0,
                    gap_extend : float = -0.5,
                    verbose    : bool  = True) -> dict:
    """
    Run the full Progressive MSA pipeline on a list of sequences.

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
        'guide_tree'        : TreeNode — root of UPGMA tree
        'dist_matrix'       : np.ndarray — pairwise distance matrix
        'runtime'           : float — total wall-clock time in seconds
    """
    if scoring_matrix is None:
        scoring_matrix = DNA_SCORE
    if labels is None:
        labels = [f"S{i}" for i in range(len(sequences))]

    m = len(sequences)
    start_time = time.time()

    # ── Step 1: Compute pairwise distance matrix ──────────────────────────
    if verbose:
        print("\n[Step 1] Computing pairwise distance matrix ...")
    dist_matrix = compute_distance_matrix(
        sequences, scoring_matrix, gap_open, gap_extend
    )
    if verbose:
        print(f"  Distance matrix shape: {dist_matrix.shape}")

    # ── Step 2: Build UPGMA guide tree ────────────────────────────────────
    if verbose:
        print("\n[Step 2] Building UPGMA guide tree ...")
    root = build_upgma_tree(dist_matrix, labels)
    if verbose:
        print("\n  Guide tree structure:")
        print_tree(root, labels, indent=1)

    # ── Step 3: Initialise one profile per leaf sequence ──────────────────
    # Also track which original sequence index maps to each profile slot
    if verbose:
        print("\n[Step 3] Initialising leaf profiles ...")
    node_profiles  = {}   # node_id -> Profile
    node_seq_order = {}   # node_id -> list of original sequence indices

    for i, seq in enumerate(sequences):
        node_profiles[i]  = sequence_to_profile(seq)
        node_seq_order[i] = [i]          # leaf tracks its own index
        if verbose:
            print(f"  Leaf {i} ({labels[i]}): {node_profiles[i]}")

    # ── Step 4: Progressive merging (post-order) ──────────────────────────
    if verbose:
        print("\n[Step 4] Progressive profile merging ...")

    merge_nodes = get_merge_order(root)

    for node in merge_nodes:
        left_id  = node.left.node_id
        right_id = node.right.node_id

        left_profile  = node_profiles[left_id]
        right_profile = node_profiles[right_id]

        if verbose:
            left_label  = (labels[node.left.seq_index]
                           if node.left.is_leaf
                           else f"cluster{left_id}")
            right_label = (labels[node.right.seq_index]
                           if node.right.is_leaf
                           else f"cluster{right_id}")
            print(f"  Merging [{left_label}] + [{right_label}] "
                  f"(dist={node.merge_dist:.4f})")

        merged_profile, merge_score = align_profiles(
            left_profile, right_profile,
            scoring_matrix, gap_open, gap_extend
        )

        if verbose:
            print(f"    → merged profile length={merged_profile.length}, "
                  f"score={merge_score:.4f}")

        node_profiles[node.node_id]  = merged_profile
        node_seq_order[node.node_id] = (node_seq_order[left_id] +
                                         node_seq_order[right_id])

    # ── Step 5: Extract final aligned sequences in original order ─────────
    final_profile = node_profiles[root.node_id]
    merge_order_indices = node_seq_order[root.node_id]
    reordered = [""] * m
    for slot, orig_idx in enumerate(merge_order_indices):
        reordered[orig_idx] = final_profile.sequences[slot]
    aligned_sequences = reordered

    # ── Step 6: Compute SP score ──────────────────────────────────────────
    if verbose:
        print("\n[Step 5] Computing Sum-of-Pairs score ...")
    sp = sum_of_pairs_score(
        aligned_sequences, scoring_matrix, gap_open, gap_extend
    )

    runtime = time.time() - start_time

    if verbose:
        print("\n" + "=" * 55)
        print("FINAL ALIGNMENT")
        print("=" * 55)
        max_label = max(len(l) for l in labels)
        for label, seq in zip(labels, aligned_sequences):
            print(f"  {label:<{max_label}} : {seq}")
        print(f"\n  Sum-of-Pairs score : {sp:.4f}")
        print(f"  Alignment length   : {len(aligned_sequences[0])}")
        print(f"  Runtime            : {runtime:.4f}s")
        print("=" * 55)

    return {
        'aligned_sequences' : aligned_sequences,
        'sp_score'          : sp,
        'guide_tree'        : root,
        'dist_matrix'       : dist_matrix,
        'runtime'           : runtime,
    }


# ─────────────────────────────────────────────
# 2.  Alignment Validation Helper
# ─────────────────────────────────────────────

def validate_alignment(aligned_sequences: list,
                       original_sequences: list) -> bool:
    """
    Sanity-check the alignment output:
        1. All aligned sequences have equal length
        2. Removing gaps from each aligned seq recovers the original
        3. No illegal characters present

    Parameters
    ----------
    aligned_sequences  : list of str — output of progressive_msa
    original_sequences : list of str — input sequences before alignment

    Returns
    -------
    bool — True if all checks pass, raises AssertionError otherwise
    """
    # Check 1: equal lengths
    lengths = set(len(s) for s in aligned_sequences)
    assert len(lengths) == 1, \
        f"INVALID: aligned sequences have unequal lengths: {lengths}"

    # Check 2: removing gaps recovers original
    for i, (aligned, original) in enumerate(
            zip(aligned_sequences, original_sequences)):
        recovered = aligned.replace('-', '')
        assert recovered == original, \
            (f"INVALID: seq {i} mismatch after gap removal\n"
             f"  Original : {original}\n"
             f"  Recovered: {recovered}")

    # Check 3: only valid characters
    valid = set('ACGT-')
    for i, seq in enumerate(aligned_sequences):
        illegal = set(seq) - valid
        assert not illegal, \
            f"INVALID: seq {i} contains illegal characters: {illegal}"

    return True


# ─────────────────────────────────────────────
# 3.  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: 3 identical sequences ─────────────────────────────────────
    print("=" * 55)
    print("Test 1: 3 identical sequences")
    seqs   = ["ACGT", "ACGT", "ACGT"]
    labels = ["Seq1", "Seq2", "Seq3"]
    result = progressive_msa(seqs, labels, verbose=True)

    assert validate_alignment(result['aligned_sequences'], seqs), \
        "Test 1 FAILED — invalid alignment"
    assert result['sp_score'] == 12.0, \
        f"Test 1 FAILED — expected SP=12.0, got {result['sp_score']}"
    print("  PASSED ✓\n")

    # ── Test 2: 4 sequences with variation ────────────────────────────────
    print("=" * 55)
    print("Test 2: 4 sequences with variation")
    seqs = [
        "ACGTACGT",
        "ACGTACGT",
        "ACGTTCGT",
        "ACGTACCT",
    ]
    labels = ["Alpha", "Beta", "Gamma", "Delta"]
    result = progressive_msa(seqs, labels, verbose=True)

    assert validate_alignment(result['aligned_sequences'], seqs), \
        "Test 2 FAILED — invalid alignment"
    print(f"  SP Score: {result['sp_score']:.4f}")
    print("  PASSED ✓\n")

    # ── Test 3: Sequences with insertions/deletions ────────────────────────
    print("=" * 55)
    print("Test 3: Sequences with different lengths (indels)")
    seqs = [
        "ACGTACGT",
        "ACGACGT",
        "ACGTCGT",
        "ACGTACGT",
    ]
    labels = ["S1", "S2", "S3", "S4"]
    result = progressive_msa(seqs, labels, verbose=True)

    assert validate_alignment(result['aligned_sequences'], seqs), \
        "Test 3 FAILED — invalid alignment"
    print(f"  SP Score: {result['sp_score']:.4f}")
    print("  PASSED ✓\n")

    # ── Test 4: 2 sequences (minimum case) ────────────────────────────────
    print("=" * 55)
    print("Test 4: 2 sequences minimum case")
    seqs   = ["ACGT", "AGGT"]
    labels = ["P", "Q"]
    result = progressive_msa(seqs, labels, verbose=False)

    assert validate_alignment(result['aligned_sequences'], seqs), \
        "Test 4 FAILED — invalid alignment"
    print(f"  Aligned: {result['aligned_sequences']}")
    print(f"  SP Score: {result['sp_score']:.4f}")
    print("  PASSED ✓\n")

    print("=" * 55)
    print("All tests passed! Module 5 is ready.")
    print("Progressive MSA pipeline is fully functional.")