# Edge coloring (cubic graphs, n=16) on the official GRAM code

Runs the official `pretrain.py` / model / loss **unchanged** on the GRAM-community
edge-coloring benchmark (3-edge-coloring of connected bridgeless cubic graphs on
16 vertices, isomorphism-disjoint split). Everything lives in this folder; no
file outside `ec16/` is modified.

| File | What it does |
|---|---|
| `build_ec16_dataset.py` | converts the community `data/edge_coloring_n16/dataset.pt` into the official PuzzleDataset format (`train/`, `test/`, `holdout/`) |
| `check_onehot.py` | preflight: can this torch build train on ignore labels (-100)? picks the dataset variant |
| `train_ec16.sh` | `scripts/graph_coloring_10.sh` with the data path swapped (official graph-coloring recipe, `kl_coef` 0.45 by default) |
| `eval_ec16.py` | loads checkpoints like `eval_pass_at_k.py`, samples with the official `sample_parallel`, scores with a real 3-edge-coloring verifier |
| `ec16_metrics.py` | verifier, conflict count, S3 x Aut(G) canonical form (same definitions as the community `eval_graph2`) |

Why `eval_pass_at_k.py` is not used: it only accepts sudoku / arc / nqueen /
graph-coloring data paths, and its graph-coloring coverage enumerates all
3^n vertex colorings. Why the data dir must not contain `graph-coloring`:
`models/losses.py` picks its verifier from the data path; with a neutral name it
falls back to exact label matching (so the in-training `exact_accuracy` /
wandb `all/exact_accuracy` is exact match against ONE stored coloring, an
underestimate of validity — use `eval_ec16.py` for real numbers).

## Mapping to the community ec-v5 setup

Same split, same 5% holdout (every 20th train graph, removed from training),
one train row per (graph, essential coloring), S3-canonical labels, loss on edge
positions only. Differences that come from the official code, not from us:
TRM-style shared network, bf16, AdamATan2 with constant LR, one optimizer step
per supervision step with ACT halting, EMA 0.999, KL summed over the 120
content tokens × 256 latent dims. ec-v5's on-the-fly vertex-relabel
augmentation is not reproduced (it was neutral in ec-v5's ablation A14).

## Run (from the repo root)

**1. Preflight** (which label variant this torch can train on):
```bash
python ec16/check_onehot.py
```
If it prints `NOT SAFE`, add `--non-edge-labels no_edge` in step 2. That variant
also supervises the 96 non-edge positions with the NO_EDGE token (trivial to
copy), so edge colors carry 24/120 of the cross-entropy instead of all of it.

**2. Build the data** (CPU, ~1 min):
```bash
python ec16/build_ec16_dataset.py --source /path/to/GRAM/data/edge_coloring_n16/dataset.pt --output-dir data/edge-coloring-16
```
The printed summary should show `test_graphs` 578, `test_essentials` 2101 and
`holdout_graphs` 164 (the ec-v5 numbers).

**3. Train** (8 GPUs; ~15 optimizer steps per epoch, 5000 epochs):
```bash
bash ec16/train_ec16.sh 2>&1 | tee ec16_train.log
```
Checkpoints land in `checkpoints/Edge-coloring-16-ACT-torch/<run_name>/step_<N>`
every 50 epochs. Different KL weight: `KL_COEF=0.2 bash ec16/train_ec16.sh`.

**4. Select a checkpoint on the holdout graphs** (never on test):
```bash
python ec16/eval_ec16.py --split holdout --num_samples 8 --checkpoint_path checkpoints/Edge-coloring-16-ACT-torch/<run_name>/step_* --output ec16_holdout.json
```
It prints every checkpoint and the best one by `valid_sample_rate`.

**5. Test once** on the selected checkpoint, and on the final one (the official
code itself does no selection):
```bash
python ec16/eval_ec16.py --split test --num_samples 20 --checkpoint_path checkpoints/Edge-coloring-16-ACT-torch/<run_name>/step_<N> --output ec16_test_step_<N>.json
```
Add `--max_h_cycles 64` for inference-depth scaling, `--no_ema` for raw weights.

## Reading the numbers against ec-v5 (seed 42, test)

| key | ec-v5 B33 |
|---|---|
| `accuracy_valid_solution` (single sample) | 0.670 |
| `pass_at_k` (k=20) | 1.000 |
| `coverage` (essential solutions) | 0.879 |
| `conflicts_single.mean` | 1.29 |
| ec-v5 `accuracy_lprm_select_at_k` ≈ here `accuracy_qhalt_select_at_k` | 1.000 |
| `accuracy_valid_solution` with 64 supervision steps | 0.817 |

`accuracy_qhalt_select_at_k` uses the official halting head as the selector; on
this data it is trained on exact match, like ec-v5's LPRM head.
