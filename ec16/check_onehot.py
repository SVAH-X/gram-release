#!/usr/bin/env python3
"""Preflight: can this torch build train on data with ignore labels (-100)?

The official model one-hot encodes the training labels for the posterior
(models/recursive_reasoning/gram.py: F.one_hot(batch["labels"], vocab_size)).
Datasets with ignore_label_id (official graph coloring, and ec16 built with the
default --non-edge-labels ignore) put -100 there. Older torch.compile lowered
one_hot to an equality test (-100 -> all-zero row); newer builds assert instead,
and training dies on the first step with a device-side assert.

Runs the same op the same way pretrain.py does (inside torch.compile, on CUDA)
in a SUBPROCESS, because a device-side assert poisons the CUDA context.
"""
import subprocess
import sys

PROBE = r"""
import torch, torch.nn.functional as F
f = torch.compile(lambda t: F.one_hot(t.long(), num_classes=6).to(torch.float32).sum(-1))
out = f(torch.tensor([[3, -100, 5]], device="cuda"))
torch.cuda.synchronize()
print("row sums:", out.cpu().tolist())
"""

if __name__ == "__main__":
    import torch
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode == 0 and "[[1.0, 0.0, 1.0]]" in r.stdout:
        print("OK: ignore labels are safe -> build with the default --non-edge-labels ignore")
        sys.exit(0)
    print((r.stderr or "").strip().splitlines()[-1:] or "")
    print("NOT SAFE: build the dataset with --non-edge-labels no_edge "
          "(official graph-coloring training would hit the same error in this environment)")
    sys.exit(1)
