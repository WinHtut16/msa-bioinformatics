"""
upgma.py
--------------------
Module 3: Guide Tree Construction using UPGMA
(Unweighted Pair Group Method with Arithmetic mean)

Algorithm:
    1. Start with each sequence as its own cluster
    2. Find the two closest clusters using a min-heap
    3. Merge them into a new cluster
    4. Update distances using UPGMA averaging formula:
           dist(new, k) = (|A| * dist(A,k) + |B| * dist(B,k)) / (|A| + |B|)
    5. Repeat until one cluster remains (the root)

The result is a binary tree where:
    - Leaf nodes = individual sequences
    - Internal nodes = merged clusters
    - Post-order traversal gives the alignment order for progressive MSA

Author: Win Htut Naing (st126687)
Course: Algorithms Design and Analysis (2026)
"""

import heapq
import numpy as np


# ─────────────────────────────────────────────
# 1.  Tree Node Structure
# ─────────────────────────────────────────────

class TreeNode:
    """
    Represents a node in the UPGMA guide tree.

    Attributes
    ----------
    node_id     : int   — unique identifier
    is_leaf     : bool  — True if this node is a sequence (leaf)
    seq_index   : int   — index into original sequences list (leaves only)
    left        : TreeNode or None
    right       : TreeNode or None
    merge_dist  : float — distance at which this node was formed
    cluster     : list  — list of original sequence indices in this cluster
    """
    def __init__(self, node_id: int, is_leaf: bool = False,
                 seq_index: int = -1):
        self.node_id    = node_id
        self.is_leaf    = is_leaf
        self.seq_index  = seq_index   # only meaningful for leaves
        self.left       = None
        self.right      = None
        self.merge_dist = 0.0
        self.cluster    = [seq_index] if is_leaf else []

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(id={self.node_id}, seq={self.seq_index})"
        return (f"Node(id={self.node_id}, "
                f"dist={self.merge_dist:.4f}, "
                f"cluster={self.cluster})")


# ─────────────────────────────────────────────
# 2.  UPGMA Core
# ─────────────────────────────────────────────

def build_upgma_tree(dist_matrix: np.ndarray,
                     labels: list = None) -> TreeNode:
    """
    Build a UPGMA guide tree from a pairwise distance matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray of shape (m, m)
                  Symmetric distance matrix from distance_matrix.py
    labels      : list of str — sequence names (optional, for printing)

    Returns
    -------
    root : TreeNode — root of the completed guide tree
    """
    m = dist_matrix.shape[0]
    if labels is None:
        labels = [f"S{i}" for i in range(m)]

    # ── Step 1: Initialise one leaf node per sequence ─────────────────────
    # active_nodes maps cluster_id -> TreeNode
    active_nodes = {}
    # cluster_sizes maps cluster_id -> number of sequences in cluster
    cluster_sizes = {}
    # dist maps (id_i, id_j) -> current distance between clusters
    dist = {}

    for i in range(m):
        node = TreeNode(node_id=i, is_leaf=True, seq_index=i)
        active_nodes[i] = node
        cluster_sizes[i] = 1

    # Populate distance lookup from the matrix
    for i in range(m):
        for j in range(m):
            dist[(i, j)] = float(dist_matrix[i][j])

    # ── Step 2: Build min-heap of (distance, id_i, id_j) ─────────────────
    # Only push pairs where i < j to avoid duplicates
    heap = []
    for i in range(m):
        for j in range(i + 1, m):
            heapq.heappush(heap, (dist[(i, j)], i, j))

    next_node_id = m          # IDs for internal nodes start after leaves
    root = None

    # ── Step 3: UPGMA merging loop ────────────────────────────────────────
    while len(active_nodes) > 1:

        # Find the closest ACTIVE pair using min-heap
        # Skip stale entries (clusters already merged)
        while heap:
            d, i, j = heapq.heappop(heap)
            if i in active_nodes and j in active_nodes:
                break
        else:
            break   # No more valid pairs

        # Create new internal node merging clusters i and j
        new_node = TreeNode(node_id=next_node_id, is_leaf=False)
        new_node.left       = active_nodes[i]
        new_node.right      = active_nodes[j]
        new_node.merge_dist = d
        new_node.cluster    = (active_nodes[i].cluster +
                                active_nodes[j].cluster)

        size_i = cluster_sizes[i]
        size_j = cluster_sizes[j]
        new_size = size_i + size_j

        print(f"  Merging cluster {i} {active_nodes[i].cluster} "
              f"+ cluster {j} {active_nodes[j].cluster} "
              f"at distance {d:.4f}")

        # Update distances from new cluster to all remaining active clusters
        for k in list(active_nodes.keys()):
            if k == i or k == j:
                continue
            # UPGMA averaging formula
            new_dist = ((size_i * dist[(i, k)] +
                         size_j * dist[(j, k)]) / new_size)
            dist[(next_node_id, k)] = new_dist
            dist[(k, next_node_id)] = new_dist
            # Push updated distance into heap
            heapq.heappush(heap, (new_dist, min(k, next_node_id),
                                              max(k, next_node_id)))

        # Remove old clusters, add new one
        del active_nodes[i]
        del active_nodes[j]
        active_nodes[next_node_id] = new_node
        cluster_sizes[next_node_id] = new_size

        root = new_node
        next_node_id += 1

    # Edge case: only one sequence
    if root is None and len(active_nodes) == 1:
        root = list(active_nodes.values())[0]

    return root


# ─────────────────────────────────────────────
# 3.  Tree Traversal Utilities
# ─────────────────────────────────────────────

def post_order_traversal(node: TreeNode) -> list:
    """
    Return list of TreeNodes in post-order (left, right, root).
    This gives us the bottom-up merge order for progressive alignment.

    Parameters
    ----------
    node : TreeNode — root of the tree

    Returns
    -------
    list of TreeNode in post-order
    """
    if node is None:
        return []
    result = []
    result += post_order_traversal(node.left)
    result += post_order_traversal(node.right)
    result.append(node)
    return result


def get_merge_order(root: TreeNode) -> list:
    """
    Return only the internal nodes in post-order.
    Each internal node represents one profile-profile alignment step.

    Returns
    -------
    list of internal TreeNodes in merge order
    """
    return [n for n in post_order_traversal(root) if not n.is_leaf]


def print_tree(node: TreeNode, labels: list = None,
               indent: int = 0, prefix: str = "Root") -> None:
    """
    Recursively print the tree structure for visual inspection.
    """
    if node is None:
        return
    pad = "    " * indent
    if node.is_leaf:
        label = labels[node.seq_index] if labels else f"Seq{node.seq_index}"
        print(f"{pad}{prefix}── {label}")
    else:
        print(f"{pad}{prefix}── [dist={node.merge_dist:.4f}] "
              f"cluster={node.cluster}")
        print_tree(node.left,  labels, indent + 1, prefix="L")
        print_tree(node.right, labels, indent + 1, prefix="R")


# ─────────────────────────────────────────────
# 4.  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: 3 sequences, known distances ──────────────────────────────
    # S0 and S1 are close (0.1), both far from S2 (0.8)
    # Expected tree: ((S0, S1), S2)
    print("=" * 55)
    print("Test 1: 3 sequences — S0,S1 close; S2 far")
    dm = np.array([
        [0.0, 0.1, 0.8],
        [0.1, 0.0, 0.8],
        [0.8, 0.8, 0.0],
    ])
    labels = ["S0", "S1", "S2"]
    root = build_upgma_tree(dm, labels)
    print("\nTree structure:")
    print_tree(root, labels)

    # Verify: root should merge S0,S1 first
    assert set(root.left.cluster + root.right.cluster) == {0, 1, 2}, \
        "Test 1 FAILED — root cluster wrong"
    # The first merge should contain S0 and S1
    first_merge = root.left if not root.left.is_leaf else root.right
    assert set(first_merge.cluster) == {0, 1}, \
        "Test 1 FAILED — S0 and S1 should merge first"
    print("  PASSED ✓\n")

    # ── Test 2: 4 sequences ────────────────────────────────────────────────
    print("=" * 55)
    print("Test 2: 4 sequences — two close pairs")
    dm = np.array([
        [0.0, 0.1, 0.9, 0.9],
        [0.1, 0.0, 0.9, 0.9],
        [0.9, 0.9, 0.0, 0.2],
        [0.9, 0.9, 0.2, 0.0],
    ])
    labels = ["S0", "S1", "S2", "S3"]
    root = build_upgma_tree(dm, labels)
    print("\nTree structure:")
    print_tree(root, labels)

    merge_order = get_merge_order(root)
    print(f"\nMerge order (post-order internal nodes):")
    for node in merge_order:
        print(f"  Node {node.node_id}: cluster={node.cluster}, "
              f"dist={node.merge_dist:.4f}")

    assert set(root.cluster) == {0, 1, 2, 3}, \
        "Test 2 FAILED — root should contain all sequences"
    print("  PASSED ✓\n")

    # ── Test 3: Post-order traversal ───────────────────────────────────────
    print("=" * 55)
    print("Test 3: Post-order traversal check")
    all_nodes = post_order_traversal(root)
    leaf_nodes = [n for n in all_nodes if n.is_leaf]
    int_nodes  = [n for n in all_nodes if not n.is_leaf]
    print(f"  Total nodes   : {len(all_nodes)}")
    print(f"  Leaf nodes    : {len(leaf_nodes)}  (expected: 4)")
    print(f"  Internal nodes: {len(int_nodes)}   (expected: 3)")
    assert len(leaf_nodes) == 4, "Test 3 FAILED — wrong leaf count"
    assert len(int_nodes)  == 3, "Test 3 FAILED — wrong internal node count"
    print("  PASSED ✓\n")

    # ── Test 4: Single sequence edge case ─────────────────────────────────
    print("=" * 55)
    print("Test 4: Single sequence edge case")
    dm = np.array([[0.0]])
    root = build_upgma_tree(dm, ["S0"])
    assert root.is_leaf and root.seq_index == 0, "Test 4 FAILED"
    print("  PASSED ✓\n")

    print("=" * 55)
    print("All tests passed! Module 3 is ready.")