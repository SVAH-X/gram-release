#!/bin/bash
# Official GRAM on edge coloring (cubic graphs, n=16).
# Identical to scripts/graph_coloring_10.sh (the official graph-coloring recipe)
# except for the data path and run name. KL_COEF defaults to the official
# graph-coloring-10 value; override with e.g. KL_COEF=0.2 bash ec16/train_ec16.sh
KL_COEF=${KL_COEF:-0.45}
run_name=${RUN_NAME:-pretrain_att_gram_edge_coloring_16_kl${KL_COEF}}
torchrun --nproc-per-node ${NPROC:-8} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=gram \
seed=42 \
global_batch_size=768 \
data_paths="[data/edge-coloring-16]" \
evaluators="[]" \
epochs=5000 eval_interval=50 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
arch.kl_balance_alpha=0.8 \
arch.kl_coef=${KL_COEF} \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
