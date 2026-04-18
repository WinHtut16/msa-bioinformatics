"""
data_generator.py
--------------------
Module 7a: Synthetic Biological Sequence Generator

Generates controlled synthetic DNA sequences for:
    - Scalability testing (vary m and l)
    - Quality comparison (vary mutation rate)
    - Reproducibility (fixed random seeds)

Generation process:
    1. Create a random base sequence of length l
    2. For each additional sequence, apply random mutations:
       - Substitutions: change a character to another
       - Insertions: insert a random character
       - Deletions: remove a character
    3. Mutation rate controls sequence divergence

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import random
import numpy as np

DNA_CHARS = ['A', 'C', 'G', 'T']


# ─────────────────────────────────────────────
# 1.  Base Sequence Generator
# ─────────────────────────────────────────────

def generate_base_sequence(length: int, seed: int = 42) -> str:
    """
    Generate a random DNA base sequence of given length.

    Parameters
    ----------
    length : int — desired sequence length
    seed   : int — random seed for reproducibility

    Returns
    -------
    str — random DNA sequence
    """
    random.seed(seed)
    return ''.join(random.choices(DNA_CHARS, k=length))


# ─────────────────────────────────────────────
# 2.  Mutate a Sequence
# ─────────────────────────────────────────────

def mutate_sequence(seq          : str,
                    mutation_rate: float = 0.1,
                    indel_rate   : float = 0.05,
                    seed         : int   = None) -> str:
    """
    Apply random mutations to a sequence to simulate evolution.

    Mutation types:
        - Substitution : replace a character with a different one
        - Insertion    : insert a random character at a position
        - Deletion     : remove a character

    Parameters
    ----------
    seq           : str   — original DNA sequence
    mutation_rate : float — probability of substitution per position
    indel_rate    : float — probability of insertion or deletion per position
    seed          : int   — random seed (None = truly random)

    Returns
    -------
    str — mutated sequence
    """
    if seed is not None:
        random.seed(seed)

    result = list(seq)
    i = 0
    while i < len(result):
        roll = random.random()

        if roll < mutation_rate:
            # Substitution: replace with a different character
            others = [c for c in DNA_CHARS if c != result[i]]
            result[i] = random.choice(others)
            i += 1

        elif roll < mutation_rate + indel_rate / 2:
            # Insertion: insert random char before position i
            result.insert(i, random.choice(DNA_CHARS))
            i += 2   # skip past inserted character

        elif roll < mutation_rate + indel_rate:
            # Deletion: remove character at position i
            result.pop(i)
            # don't increment i — next char slides into position

        else:
            i += 1

    return ''.join(result)


# ─────────────────────────────────────────────
# 3.  Generate a Full Sequence Set
# ─────────────────────────────────────────────

def generate_sequences(m            : int,
                       l            : int,
                       mutation_rate: float = 0.1,
                       indel_rate   : float = 0.05,
                       seed         : int   = 42) -> tuple:
    """
    Generate m homologous DNA sequences of approximate length l.

    Parameters
    ----------
    m             : int   — number of sequences
    l             : int   — base sequence length
    mutation_rate : float — substitution rate per position
    indel_rate    : float — indel rate per position
    seed          : int   — master random seed

    Returns
    -------
    (sequences, labels) : tuple
        sequences : list of str — m mutated sequences
        labels    : list of str — sequence names
    """
    base = generate_base_sequence(l, seed=seed)
    sequences = [base]   # first sequence is the base (no mutation)
    labels    = [f"Seq_0"]

    for i in range(1, m):
        mutated = mutate_sequence(
            base,
            mutation_rate=mutation_rate,
            indel_rate=indel_rate,
            seed=seed + i   # different seed per sequence
        )
        sequences.append(mutated)
        labels.append(f"Seq_{i}")

    return sequences, labels


# ─────────────────────────────────────────────
# 4.  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("Test 1: Base sequence generation")
    seq = generate_base_sequence(20, seed=42)
    print(f"  Base sequence (l=20): {seq}")
    assert len(seq) == 20, "Test 1 FAILED"
    assert all(c in DNA_CHARS for c in seq), "Test 1 FAILED — illegal chars"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("Test 2: Mutation with low rate")
    base    = "ACGTACGTACGT"
    mutated = mutate_sequence(base, mutation_rate=0.1,
                               indel_rate=0.05, seed=99)
    print(f"  Original : {base}")
    print(f"  Mutated  : {mutated}")
    assert all(c in DNA_CHARS for c in mutated), \
        "Test 2 FAILED — illegal chars"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("Test 3: Higher mutation rate produces more differences")
    base     = generate_base_sequence(50, seed=1)
    low_mut  = mutate_sequence(base, mutation_rate=0.05,
                                indel_rate=0.01, seed=10)
    high_mut = mutate_sequence(base, mutation_rate=0.40,
                                indel_rate=0.10, seed=10)
    diff_low  = sum(a != b for a, b in zip(base, low_mut[:len(base)]))
    diff_high = sum(a != b for a, b in zip(base, high_mut[:len(base)]))
    print(f"  Low mutation differences  : {diff_low}")
    print(f"  High mutation differences : {diff_high}")
    assert diff_high >= diff_low, \
        "Test 3 FAILED — higher rate should produce more differences"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("Test 4: Generate sequence set")
    seqs, labels = generate_sequences(m=5, l=30,
                                       mutation_rate=0.1,
                                       indel_rate=0.05,
                                       seed=42)
    print(f"  Generated {len(seqs)} sequences:")
    for lbl, seq in zip(labels, seqs):
        print(f"    {lbl} (len={len(seq)}): {seq}")
    assert len(seqs) == 5, "Test 4 FAILED — wrong number of sequences"
    assert seqs[0] == generate_base_sequence(30, seed=42), \
        "Test 4 FAILED — first sequence should be unmodified base"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("All tests passed! data_generator.py is ready.")