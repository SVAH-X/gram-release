"""Edge-coloring (cubic graphs, n=16) scoring helpers for ec16/eval_ec16.py.

Pure-python/numpy re-statement of the GRAM-community `gram/graph_tasks`
verifier, conflict counter and S3 x Aut(G) canonicalization (MIT), so the
scores here are directly comparable with that repo's `eval_graph2` numbers
(accuracy_valid_solution, pass_at_k, coverage, conflicts).

Token layout (identical to the source dataset):
  position k <-> vertex pair (i, j), i < j, in lexicographic order over a
  fixed max_n x max_n upper triangle (seq_len = C(max_n, 2) = 120 for n=16).
  Vocab: 0 PAD | 1 NO_EDGE | 2 EDGE (input only) | 3/4/5 = colors 0/1/2.

Only the EDGE positions of a prediction are read; every other position is
ignored (the training labels mask them too).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

Edge = Tuple[int, int]

PAD, NO_EDGE, EDGE, COLOR0 = 0, 1, 2, 3
N_COLORS = 3


@lru_cache(maxsize=None)
def layout(max_n: int) -> Tuple[Tuple[Edge, ...], Dict[Edge, int]]:
    pairs = tuple((i, j) for i in range(max_n) for j in range(i + 1, max_n))
    return pairs, {p: k for k, p in enumerate(pairs)}


def max_n_from_seq_len(seq_len: int) -> int:
    n = 1
    while n * (n - 1) // 2 < seq_len:
        n += 1
    if n * (n - 1) // 2 != seq_len:
        raise ValueError(f"seq_len {seq_len} is not C(n, 2) for any n")
    return n


def decode_graph(tokens: Sequence[int], max_n: int) -> Tuple[int, List[Edge]]:
    """(n, edges) from an input sequence; edges in layout order."""
    pairs, _ = layout(max_n)
    n, edges = 0, []
    for k, (i, j) in enumerate(pairs):
        tok = int(tokens[k])
        if tok == PAD:
            continue
        n = max(n, j + 1)
        if tok != NO_EDGE:
            edges.append((i, j))
    return n, edges


def decode_colors(tokens: Sequence[int], edges: Sequence[Edge], max_n: int) -> List[int]:
    """Color per edge (0..2); a non-color token at an edge position -> -1."""
    _, idx = layout(max_n)
    out = []
    for e in edges:
        tok = int(tokens[idx[e]])
        out.append(tok - COLOR0 if COLOR0 <= tok < COLOR0 + N_COLORS else -1)
    return out


def _incidence(n: int, edges: Sequence[Edge]) -> List[List[int]]:
    inc: List[List[int]] = [[] for _ in range(n)]
    for ei, (u, v) in enumerate(edges):
        inc[u].append(ei)
        inc[v].append(ei)
    return inc


def is_valid(n: int, edges: Sequence[Edge], colors: Sequence[int]) -> bool:
    """Proper 3-edge-coloring: colors in {0,1,2} and all distinct at every vertex."""
    if any(c not in (0, 1, 2) for c in colors):
        return False
    for eis in _incidence(n, edges):
        seen = [colors[i] for i in eis]
        if len(seen) != len(set(seen)):
            return False
    return True


def conflicts(n: int, edges: Sequence[Edge], colors: Sequence[int]) -> int:
    """Incident edge pairs with the same (real) color. Random baseline for cubic n: n."""
    c = 0
    for eis in _incidence(n, edges):
        for a in range(len(eis)):
            for b in range(a + 1, len(eis)):
                ca, cb = colors[eis[a]], colors[eis[b]]
                if ca >= 0 and ca == cb:
                    c += 1
    return c


def edge_automorphisms(n: int, edges: Sequence[Edge]) -> List[List[int]]:
    """Aut(G) as permutations of edge indices (p[i] = index of edge i's image)."""
    import networkx as nx

    index = {e: i for i, e in enumerate(edges)}
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    perms: List[List[int]] = []
    for iso in nx.algorithms.isomorphism.GraphMatcher(g, g).isomorphisms_iter():
        p = [0] * len(edges)
        for i, (u, v) in enumerate(edges):
            a, b = iso[u], iso[v]
            p[i] = index[(a, b) if a < b else (b, a)]
        perms.append(p)
    return perms


def canonical_coloring(colors: Sequence[int], edge_perms: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """Lex-smallest coloring over the orbit under S3 x Aut(G) (= one 'essential' solution)."""
    best = None
    m = len(colors)
    for p in edge_perms:
        permuted = [0] * m
        for i in range(m):
            permuted[p[i]] = colors[i]
        for cp in itertools.permutations(range(N_COLORS)):
            cand = tuple(cp[x] for x in permuted)
            if best is None or cand < best:
                best = cand
    return best if best is not None else tuple(colors)


def s3_canonicalize(tokens: Sequence[int]) -> List[int]:
    """First-appearance color relabeling (structural tokens untouched).

    Same convention as GRAM-community `relabel.s3_canonicalize_tokens` and as the
    official graph-coloring data ("canonical colorings", first appearance order).
    """
    remap: Dict[int, int] = {}
    for tok in tokens:
        tok = int(tok)
        if tok >= COLOR0 and tok not in remap:
            remap[tok] = COLOR0 + len(remap)
            if len(remap) == N_COLORS:
                break
    for c in range(COLOR0, COLOR0 + N_COLORS):
        if c not in remap:
            remap[c] = COLOR0 + len(remap)
    return [remap.get(int(t), int(t)) if int(t) >= COLOR0 else int(t) for t in tokens]
