#!/usr/bin/env python3
"""Convert the GRAM-community edge-coloring dataset (cubic graphs, n=16) into the
official PuzzleDataset format, so pretrain.py / the official model run on it
UNCHANGED.

Source: `data/edge_coloring_n16/dataset.pt` written by the GRAM-community
`scripts/build_edge_coloring_n16.py` (format edge_coloring_orbit_v1).

What is reproduced from that repo's ec-v5 setup:
  * the same isomorphism-disjoint train/test split (graph ids stored in dataset.pt);
  * the same held-out slice: every 20th train graph (holdout_frac 0.05) is REMOVED
    from train and written as its own split `holdout/` (for checkpoint selection,
    so test is never used for selection);
  * one-to-many supervision: one train row per (graph, essential coloring) pair,
    labels S3-canonicalized by first appearance (same as ec-v5 and as the official
    graph-coloring data);
  * the loss sees colors only at EDGE positions: every other label position is 0,
    which is the dataset's ignore_label_id (= ec-v5's edge-masked CE). This is the
    official graph-coloring convention too, and it means training labels contain
    IGNORE_LABEL_ID (-100), which the posterior feeds to F.one_hot. Recent torch
    versions assert on negative one_hot inputs (also under torch.compile), which
    would crash official graph-coloring training as well. If that happens in your
    environment, rebuild with --non-edge-labels no_edge: non-edge positions then
    carry the NO_EDGE token (1) as a supervised label, so no -100 appears, at the
    cost of the CE also averaging over 96 trivially-copyable positions.
Not reproduced (the official trainer has no hook for it): on-the-fly vertex-relabel
augmentation (ec-v5 ablation A14 found it neutral).

Format details follow dataset/build_graph_coloring_dataset.py: each row is its own
puzzle and group, puzzle_identifiers all 0, vocab 6, pad_id = ignore_label_id = 0.
test/ and holdout/ hold ONE row per graph (label = its first essential coloring,
only used by the in-training exact-match metric; validity is scored by
ec16/eval_ec16.py).

The output directory name must NOT contain "nqueen", "graph-coloring",
"graph_coloring" or "sudoku": models/losses.py picks its task verifier from the
data path, and none of those verifiers is an edge-coloring verifier. With a
neutral name the official code falls back to exact label matching.

Usage (from the repo root):
    python ec16/build_ec16_dataset.py \
        --source /path/to/GRAM/data/edge_coloring_n16/dataset.pt \
        --output-dir data/edge-coloring-16
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ec16_metrics as em  # noqa: E402

FORBIDDEN = ("nqueen", "graph-coloring", "graph_coloring", "sudoku")


def _write_split(out_dir: str, split: str, inputs: np.ndarray, labels: np.ndarray, vocab_size: int) -> None:
    n_rows, seq_len = inputs.shape
    save_dir = os.path.join(out_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    arrays = {
        "inputs": inputs.astype(np.uint8),
        "labels": labels.astype(np.uint8),
        "puzzle_identifiers": np.zeros(n_rows, dtype=np.int32),
        "puzzle_indices": np.arange(n_rows + 1, dtype=np.int32),
        "group_indices": np.arange(n_rows + 1, dtype=np.int32),
    }
    for k, v in arrays.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    # Same fields as dataset/common.py::PuzzleDatasetMetadata.
    meta = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_puzzle_identifiers": 1,
        "total_groups": n_rows,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": n_rows,
        "sets": ["all"],
    }
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(meta, f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="GRAM-community data/edge_coloring_n16/dataset.pt")
    ap.add_argument("--output-dir", default="data/edge-coloring-16")
    ap.add_argument("--holdout-frac", type=float, default=0.05,
                    help="every round(1/frac)-th train graph goes to holdout/ (ec-v5: 0.05)")
    ap.add_argument("--non-edge-labels", choices=["ignore", "no_edge"], default="ignore",
                    help="label at non-edge positions: 'ignore' (masked out of the loss, = ec-v5 "
                         "and the official graph-coloring convention) or 'no_edge' (supervised "
                         "NO_EDGE token; fallback when torch rejects -100 in F.one_hot)")
    args = ap.parse_args()

    lowered = args.output_dir.lower()
    bad = [w for w in FORBIDDEN if w in lowered]
    if bad:
        raise SystemExit(f"--output-dir {args.output_dir!r} contains {bad}; models/losses.py would "
                         f"apply the wrong task verifier. Use e.g. data/edge-coloring-16.")

    d = torch.load(args.source, map_location="cpu", weights_only=False)
    if d.get("format") != "edge_coloring_orbit_v1" or d.get("task") != "edge_coloring":
        raise SystemExit(f"unexpected source format/task: {d.get('format')!r} / {d.get('task')!r}")

    max_n = int(d["max_n"])
    seq_len = int(d["seq_len"])
    vocab_size = int(d["vocab_size"])
    assert seq_len == max_n * (max_n - 1) // 2, (seq_len, max_n)

    input_tokens = d["input_tokens"].long().numpy()          # (G, L)
    target_values = d["target_values"].long().numpy()        # (T, L)
    target_offsets = d["target_offsets"].long().tolist()     # (G+1,)
    edges_values = d["edges_values"].long().tolist()
    edges_offsets = d["edges_offsets"].long().tolist()
    n_solutions = d["n_solutions"].long().tolist()
    test_ids = [int(g) for g in d["test_graph_ids"].tolist()]
    pair_graph = [int(g) for g in d["train_pair_graph"].tolist()]
    pair_target = [int(t) for t in d["train_pair_target"].tolist()]

    # Holdout exactly as GRAM-community gram/datasets/edge_coloring_v2.py.
    train_graph_ids = sorted(set(pair_graph))
    holdout_ids: set[int] = set()
    if args.holdout_frac > 0:
        step = max(2, round(1.0 / args.holdout_frac))
        holdout_ids = set(train_graph_ids[::step])

    def label_of(target_row: np.ndarray) -> np.ndarray:
        lab = np.asarray(em.s3_canonicalize(target_row.tolist()), dtype=np.int64)
        if args.non_edge_labels == "ignore":
            lab[lab < em.COLOR0] = 0      # supervise colors at edge positions only
        else:
            assert set(np.unique(lab[lab < em.COLOR0]).tolist()) <= {em.NO_EDGE}, \
                "n < max_n graphs (PAD positions) are not supported with --non-edge-labels no_edge"
        return lab

    def graph_record(g: int) -> dict:
        edges = [tuple(e) for e in edges_values[edges_offsets[g]:edges_offsets[g + 1]]]
        n, dec_edges = em.decode_graph(input_tokens[g].tolist(), max_n)
        assert dec_edges == sorted(edges), f"graph {g}: input tokens disagree with stored edges"
        return {"graph_id": g, "n": n, "edges": [list(e) for e in dec_edges],
                "n_solutions": int(n_solutions[g]),
                "input_tokens": input_tokens[g].tolist()}

    # ---- train: one row per (graph, essential coloring), holdout graphs removed ----
    tr_in, tr_lab = [], []
    for g, t in zip(pair_graph, pair_target):
        if g in holdout_ids:
            continue
        lab = label_of(target_values[t])
        # sanity: the label is a valid coloring of this graph
        n, edges = em.decode_graph(input_tokens[g].tolist(), max_n)
        assert em.is_valid(n, edges, em.decode_colors(lab.tolist(), edges, max_n)), (g, t)
        tr_in.append(input_tokens[g])
        tr_lab.append(lab)

    # ---- test / holdout: one row per graph ----
    def unique_split(ids):
        rows_in, rows_lab, records = [], [], []
        for g in ids:
            rows_in.append(input_tokens[g])
            rows_lab.append(label_of(target_values[target_offsets[g]]))
            records.append(graph_record(g))
        return np.stack(rows_in), np.stack(rows_lab), records

    hold_ids = sorted(holdout_ids)
    te_in, te_lab, te_rec = unique_split(test_ids)
    ho_in, ho_lab, ho_rec = unique_split(hold_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    _write_split(args.output_dir, "train", np.stack(tr_in), np.stack(tr_lab), vocab_size)
    _write_split(args.output_dir, "test", te_in, te_lab, vocab_size)
    _write_split(args.output_dir, "holdout", ho_in, ho_lab, vocab_size)
    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    # graph metadata for scoring (input tokens -> graph, edges, #essential solutions)
    for split, recs in (("test", te_rec), ("holdout", ho_rec)):
        with open(os.path.join(args.output_dir, split, "ec16_graphs.json"), "w") as f:
            json.dump({"max_n": max_n, "graphs": recs}, f)

    n_train_graphs = len(set(pair_graph) - holdout_ids)
    summary = {
        "source": os.path.abspath(args.source),
        "non_edge_labels": args.non_edge_labels,
        "max_n": max_n, "seq_len": seq_len, "vocab_size": vocab_size,
        "train_rows": len(tr_in), "train_graphs": n_train_graphs,
        "holdout_graphs": len(hold_ids),
        "holdout_essentials": int(sum(r["n_solutions"] for r in ho_rec)),
        "test_graphs": len(test_ids),
        "test_essentials": int(sum(r["n_solutions"] for r in te_rec)),
        "optimizer_steps_per_epoch_at_global_batch_768": len(tr_in) // 768,
    }
    with open(os.path.join(args.output_dir, "ec16_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
