#!/usr/bin/env python3
"""Validity / coverage evaluation of official-GRAM checkpoints on edge coloring.

Why a separate script: eval_pass_at_k.py only knows sudoku / arc / nqueen /
graph-coloring paths, and its graph-coloring coverage enumerates all 3^n vertex
colorings (n = number of graph vertices), which is a different task anyway. This script:

  * loads the checkpoint EXACTLY like eval_pass_at_k.py (same model + loss head,
    `_orig_mod.` prefix stripped, EMA weights applied when present);
  * draws K samples per graph with the official evaluators/sampling_strategies.py
    `sample_parallel` (argmax tokens, stochasticity from the latent prior, all
    `halt_max_steps` supervision steps);
  * scores every draw with a real 3-edge-coloring verifier (ec16/ec16_metrics.py),
    reporting the same quantities as the GRAM-community `eval_graph2`, under the
    same JSON key names (draw 0 serves as the single-sample draw):

      accuracy_valid_solution      draw 0 is a valid coloring (single-sample accuracy)
      pass_at_k                    any of the K draws is valid
      accuracy_majority_select_at_k  most frequent exact draw (ties -> first) is valid
      accuracy_qhalt_select_at_k   draw with the highest q_halt logit is valid
                                   (the official halting head as a value scorer)
      coverage                     distinct essential (S3 x Aut(G)) solutions found /
                                   total essential solutions
      valid_sample_rate            valid draws / (graphs x K)   <- low-noise selection key
      conflicts_single.mean        same-colored incident edge pairs, draw 0 (random = 16)
      conflicts_per_sample_mean    same, averaged over all K draws
      exact_match_rate             draw 0 equals the stored label (what the official
                                   in-training `exact_accuracy` measures here)

Several checkpoints can be passed; with --split holdout this is the checkpoint
selection pass (pick the best valid_sample_rate), then run --split test once on
the chosen checkpoint.

Usage (from the repo root, one GPU is plenty):
    python ec16/eval_ec16.py --data_path data/edge-coloring-16 --split test \
        --checkpoint_path checkpoints/Edge-coloring-16-ACT-torch/<run>/step_<N> \
        --num_samples 20 --output ec16_test_step_<N>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # repo root: pretrain.py, models/, dataset/, evaluators/

# These names must live in __main__: checkpoints pickle their config as
# `__main__.PretrainConfig` (pretrain.py runs as a script). Same trick as eval_pass_at_k.py.
from pretrain import PretrainConfig, ArchConfig, LossConfig, EvaluatorConfig  # noqa: E402,F401
from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402
from evaluators.sampling_strategies import sample_parallel  # noqa: E402
from models.ema import EMAHelper  # noqa: E402
from utils.functions import load_model_class  # noqa: E402

import ec16_metrics as em  # noqa: E402


def load_model(checkpoint_path: str, eval_metadata, device, use_ema: bool, max_h_cycles):
    ckpt = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"), map_location=device, weights_only=False)
    config = ckpt["config"]
    if max_h_cycles is not None:
        config.arch.halt_max_steps = max_h_cycles
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=eval_metadata.vocab_size,
        seq_len=eval_metadata.seq_len,
        num_puzzle_identifiers=eval_metadata.num_puzzle_identifiers,
        causal=False,
    )
    with torch.device(device):
        model = load_model_class(config.arch.name)(model_cfg)
        model = load_model_class(config.arch.loss.name)(model, **config.arch.loss.__pydantic_extra__)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, assign=True)
    used_ema = False
    if use_ema and "ema" in ckpt:
        helper = EMAHelper(mu=0.9999)            # mu is irrelevant for loading
        helper.register(model)
        helper.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()})
        helper.ema(model)
        used_ema = True
    model.eval()
    model = model.to(device)
    return model, config, used_ema, int(ckpt.get("step", -1))


class GraphBook:
    """input tokens -> graph record; caches Aut(G) per graph."""

    def __init__(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.max_n = int(d["max_n"])
        self.by_key = {}
        for r in d["graphs"]:
            r["edges"] = [tuple(e) for e in r["edges"]]
            self.by_key[bytes(np.asarray(r["input_tokens"], dtype=np.uint8))] = r
        self._aut = {}

    def lookup(self, inp_row: np.ndarray):
        return self.by_key.get(bytes(inp_row.astype(np.uint8)))

    def aut(self, r):
        g = r["graph_id"]
        if g not in self._aut:
            self._aut[g] = em.edge_automorphisms(r["n"], r["edges"])
        return self._aut[g]


@torch.inference_mode()
def evaluate(model, dataset, book: GraphBook, num_samples: int, device, data_path: str) -> dict:
    K = num_samples
    seen = set()
    tot = dict(graphs=0, valid1=0, pass_k=0, maj=0, qhalt=0, cov_found=0, ess=0,
               valid_draws=0, conf1=0, conf_all=0, exact1=0, noncolor1=0)
    conf1_list = []
    for _set, batch, _gbs in dataset:
        batch = {k: v.to(device) for k, v in batch.items()}
        samples = sample_parallel(model, batch, K, {"preds", "q_halt_logits"}, device,
                                  extract_h_states=False, data_path=data_path)
        inputs = batch["inputs"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        preds = np.stack([s["preds"].cpu().numpy() for s in samples], axis=1)           # (B, K, L)
        qh = np.stack([s["q_halt_logits"].float().cpu().numpy() for s in samples], axis=1)  # (B, K)
        for b in range(inputs.shape[0]):
            r = book.lookup(inputs[b])
            if r is None:                       # padding row of the last batch
                continue
            if r["graph_id"] in seen:
                raise RuntimeError(f"graph {r['graph_id']} seen twice")
            seen.add(r["graph_id"])
            n, edges = r["n"], r["edges"]
            valid, keys = [], set()
            for k in range(K):
                colors = em.decode_colors(preds[b, k], edges, book.max_n)
                ok = em.is_valid(n, edges, colors)
                valid.append(ok)
                c = em.conflicts(n, edges, colors)
                tot["conf_all"] += c
                if k == 0:
                    tot["conf1"] += c
                    conf1_list.append(c)
                    tot["noncolor1"] += sum(1 for x in colors if x < 0)
                if ok:
                    keys.add(em.canonical_coloring(colors, book.aut(r)))
            mask = labels[b] != -100
            tot["exact1"] += int(np.array_equal(preds[b, 0][mask], labels[b][mask]))
            tot["graphs"] += 1
            tot["valid1"] += int(valid[0])
            tot["pass_k"] += int(any(valid))
            tot["valid_draws"] += sum(valid)
            tot["cov_found"] += len(keys)
            tot["ess"] += r["n_solutions"]
            # majority: most frequent exact draw, ties -> first seen
            counts, first = {}, {}
            for k in range(K):
                t = preds[b, k].tobytes()
                counts[t] = counts.get(t, 0) + 1
                first.setdefault(t, k)
            best = max(counts, key=lambda t: (counts[t], -first[t]))
            tot["maj"] += int(valid[first[best]])
            tot["qhalt"] += int(valid[int(np.argmax(qh[b]))])
    G = max(tot["graphs"], 1)
    expected = len(book.by_key)
    if tot["graphs"] != expected:
        raise RuntimeError(f"scored {tot['graphs']} graphs but the split has {expected}")
    return {
        "n_examples": tot["graphs"],
        "n_samples": K,
        "accuracy_valid_solution": tot["valid1"] / G,
        "pass_at_k": tot["pass_k"] / G,
        "accuracy_majority_select_at_k": tot["maj"] / G,
        "accuracy_qhalt_select_at_k": tot["qhalt"] / G,
        "coverage": tot["cov_found"] / max(tot["ess"], 1),
        "coverage_found": tot["cov_found"],
        "coverage_total_essential": tot["ess"],
        "valid_sample_rate": tot["valid_draws"] / (G * K),
        "conflicts_single": {"mean": tot["conf1"] / G, "median": float(np.median(conf1_list)),
                             "min": int(min(conf1_list)), "max": int(max(conf1_list))},
        "conflicts_per_sample_mean": tot["conf_all"] / (G * K),
        "noncolor_edges_single_mean": tot["noncolor1"] / G,
        "exact_match_rate": tot["exact1"] / G,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_path", nargs="+", required=True,
                    help="one or more step_<N> directories (each holding checkpoint.pt)")
    ap.add_argument("--data_path", default="data/edge-coloring-16")
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64, help="graphs per forward (x num_samples rows)")
    ap.add_argument("--max_h_cycles", type=int, default=None,
                    help="override halt_max_steps = supervision steps at inference (depth scaling)")
    ap.add_argument("--no_ema", action="store_true", help="evaluate raw weights instead of EMA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None, help="JSON file (list when several checkpoints)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book = GraphBook(os.path.join(args.data_path, args.split, "ec16_graphs.json"))

    results = []
    for ck in args.checkpoint_path:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        dataset = PuzzleDataset(PuzzleDatasetConfig(
            seed=args.seed, dataset_paths=[args.data_path], global_batch_size=args.batch_size,
            test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1), split=args.split)
        model, config, used_ema, step = load_model(ck, dataset.metadata, device,
                                                  not args.no_ema, args.max_h_cycles)
        t0 = time.time()
        res = evaluate(model, dataset, book, args.num_samples, device, args.data_path)
        res.update({"checkpoint": ck, "step": step, "split": args.split, "seed": args.seed,
                    "used_ema": used_ema, "halt_max_steps": int(config.arch.halt_max_steps),
                    "kl_coef": float(config.arch.__pydantic_extra__.get("kl_coef", float("nan"))),
                    "seconds": round(time.time() - t0, 1)})
        results.append(res)
        print(f"[{args.split}] {ck}: valid@1={res['accuracy_valid_solution']:.4f} "
              f"valid_rate={res['valid_sample_rate']:.4f} pass@{args.num_samples}={res['pass_at_k']:.4f} "
              f"qhalt@{args.num_samples}={res['accuracy_qhalt_select_at_k']:.4f} "
              f"coverage={res['coverage']:.4f} conflicts1={res['conflicts_single']['mean']:.3f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(results) > 1:
        best = max(results, key=lambda r: (r["valid_sample_rate"], r["step"]))
        print(f"\nbest by valid_sample_rate on {args.split}: {best['checkpoint']} "
              f"({best['valid_sample_rate']:.4f})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
