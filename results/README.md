# Results

Runs of this repository's code (unchanged) on N-Queens 8×8, N-Queens 10×10 and
the GRAM-community edge-coloring benchmark (EC-16: 3-edge-coloring of cubic
graphs on 16 vertices), next to the GRAM-community implementation
("our version"). All runs: seed 42, 16 supervision steps, 20 samples per puzzle.

## Official code

8 GPUs, global batch 768, bf16, EMA weights for evaluation.

| | NQ8 | NQ10 | EC-16, KL 0.05 | EC-16, KL 0.1 | **EC-16, KL 0.2 ★** | EC-16, KL 0.45 |
|---|---|---|---|---|---|---|
| KL weight | 0.07 | 0.045 | 0.05 | 0.1 | 0.2 | 0.45 |
| Train set | 7,224 rows | 108,234 rows | 11,751 rows | 11,751 rows | 11,751 rows | 11,751 rows |
| Test set | 772 puzzles | 6,513 puzzles | 578 graphs | 578 graphs | 578 graphs | 578 graphs |
| Epochs | 5,000 | 1,000 | 5,000 | 5,000 | 5,000 | 5,000 |
| Checkpoint | final | final | best on holdout | best on holdout | best on holdout | best on holdout |
| **Accuracy** | **99.2** | **90.3** | 71.8 | 79.9 | **83.9** | 21.8 |
| **Coverage@20** | **98.3** | **90.6**† | 92.8 | **93.7** | 86.0 | 9.3 |
| Paper (accuracy / coverage) | 99.7 / 90.3 | 89.7 / 57.5 | — | — | — | — |

★ The KL value and the checkpoint were both chosen on 164 held-out training
graphs; the test set was scored once per run.
† The official evaluator checks each NQ10 puzzle against only its first 20
solutions (some have up to 93), so this coverage is approximate.

More EC-16 numbers (test set):

| | KL 0.05 | KL 0.1 | KL 0.2 | KL 0.45 |
|---|---|---|---|---|
| Checkpoint (step) | 15,300 | 26,010 | 23,715 | 76,500 (final) |
| Accuracy at 64 supervision steps | 81.1 | 84.4 | 89.8 | 23.2 |
| Coverage@20 at 64 supervision steps | 93.1 | 93.2 | 82.7 | 9.1 |
| Final checkpoint, no selection (accuracy / coverage) | — | 47.4 / 85.7 | — | 21.8 / 9.3 |

## Our version (GRAM-community implementation)

Separate low/high-level networks, AdamW with cosine LR, raw weights for evaluation.

| | NQ8 (v13) | NQ10 (v13 recipe) | EC-16 (ec-v5) |
|---|---|---|---|
| KL weight | 0.07 | 0.045 | 0.243 (B33) |
| Train set | 7,261 rows | 107,815 rows | 11,751 rows |
| Test set | 773 puzzles | 6,513 puzzles | 578 graphs |
| Epochs used | 300 of 3,000 | 60 of 800 | 650 of 1,500 |
| Checkpoint | best on test split | best on test split | best on holdout |
| **Accuracy** | **95.0** | **59.3** | **67.0** |
| **Coverage@20** | **97.2** | **83.3** | **87.9** |
| Paper (accuracy / coverage) | 99.7 / 90.3 | 89.7 / 57.5 | — |

## Definitions and caveats

- **Accuracy** is single-sample accuracy (the paper's Table 1 metric). Official
  N-Queens: `NQueens/exact_pass@1`, the first of the 20 draws on each test row.
  Official EC-16: `accuracy_valid_solution` from `ec16/eval_ec16.py`. Ours:
  NQ8/NQ10 averaged over the 20 draws, EC-16 one draw.
- **Coverage@20** is distinct valid solutions found in 20 samples ÷ total
  solutions, counting each puzzle once. For EC-16, colorings that differ only by
  renaming colors or by a graph symmetry count as one solution. The official
  N-Queens evaluator only prints coverage per number-of-solutions bucket; the
  numbers above combine those buckets (plain mean over test rows, which equals
  "each puzzle once" because a k-solution puzzle has k rows).
- **N-Queens splits differ between the two codebases.** Both build the same pool
  (NQ8: 8,464 puzzle–solution pairs / 5,148 puzzles; NQ10: 126,700 / 43,420)
  and hold out 15% of puzzles, but with different shuffles (numpy vs Python
  `random`, both seed 42) and rounding, so the test puzzles are not the same.
  The EC-16 split is identical in both.
- The official N-Queens validity check does not verify that the given queens
  were kept; ours does.

## Layout

- `runs/<project>/<run>/all_config.yaml`: resolved training config of each run.
- `runs/<project>/<run>/step_<N>/`: official N-Queens evaluator outputs
  (`eval_results_*.txt`, `nqueens_results.json`).
- `logs/`: console logs and JSON outputs of every training and evaluation run
  (EC-16 selection: `ec16_*_holdout.json`; EC-16 test: `ec16_*_test*.json`;
  timing runs: `logs/timing/`).
- `nqueens_viz/`: sample N-Queens predictions from the NQ8 evaluation.
- `ec16_data_summary.json`: EC-16 split sizes from `ec16/build_ec16_dataset.py`.

Checkpoints are not included. Training curves are on wandb
(entity `k2peng-university-of-waterloo`, projects `Nqueens-10-ACT-torch`,
`Edge-coloring-16-ACT-torch`).
