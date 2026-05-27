"""Kaggle notebook template for Stage 1 training on 2x T4 GPU.

Copy each block between '═══════' markers into a SEPARATE notebook cell.

Three things to edit in CELL 2:
  - MODEL_INPUT_DIR  : path to MentalRoBERTa model (find via CELL 1)
  - PREP_DATASET     : path to preprocessed Kaggle Dataset
  - CODE_DATASET     : path to stage1 code Kaggle Dataset
  - FOLD_THIS_NOTEBOOK : which fold to train (one per notebook; see plan below)

Notebook settings (right sidebar before "Save Version"):
  - Accelerator:    GPU T4 x2
  - Internet:       Off
  - Persistence:    No persistence (faster startup)
  - Environment:    "Pin to original environment" (avoid surprise upgrades)

Multi-notebook plan (each notebook ~2-3h actual training with current config):
  Notebook 1: fold 0
  Notebook 2: fold 1
  Notebook 3: fold 2
  Notebook 4: --mode full   (final model on full train, predicts val + test)

After all 4 notebooks finish: download outputs/predictions/ from each, merge
locally under ver2/stage1/outputs/predictions/, then:

    cd ver2/stage1
    python predict.py --assemble_oof    # -> p_text_oof_train.parquet
    python predict.py --assemble_test   # -> p_text_test_ensemble.parquet

These two parquets are the input for Stage 2 (LightGBM).
"""


# ═════════════════════════════════════════════════════════════════════════
# CELL 1 -- Discover paths (run this FIRST, copy outputs into CELL 2)
# ═════════════════════════════════════════════════════════════════════════
"""
import subprocess

print("=== Available inputs ===")
print(subprocess.run(["ls", "/kaggle/input/"], capture_output=True, text=True).stdout)

print("=== MentalRoBERTa config.json location (parent dir is MODEL_INPUT_DIR) ===")
print(subprocess.run(["find", "/kaggle/input/", "-name", "config.json"],
                     capture_output=True, text=True).stdout)

print("=== Preprocessed parquets (parent dir is PREP_DATASET) ===")
print(subprocess.run(["find", "/kaggle/input/", "-name", "posts_clean.parquet"],
                     capture_output=True, text=True).stdout)
print(subprocess.run(["find", "/kaggle/input/", "-name", "splits.parquet"],
                     capture_output=True, text=True).stdout)

print("=== Stage 1 code (parent dir is CODE_DATASET) ===")
print(subprocess.run(["find", "/kaggle/input/", "-name", "train.py"],
                     capture_output=True, text=True).stdout)
"""


# ═════════════════════════════════════════════════════════════════════════
# CELL 2 -- Configure paths and install peft
# ═════════════════════════════════════════════════════════════════════════
"""
import os, sys, shutil
from pathlib import Path

# ▼▼▼ EDIT THESE 5 VARIABLES BASED ON CELL 1 OUTPUT ▼▼▼
MODEL_INPUT_DIR = "/kaggle/input/metal/transformers/metal/1"           # contains config.json
PREP_DATA_DIR   = "/kaggle/input/dede-preprocessed"                    # contains posts_clean.parquet
CODE_DIR        = "/kaggle/input/dede-stage1-code"                     # contains train.py
FOLD_THIS_NOTEBOOK = 0                                                 # which fold (0..2); ignored for full mode
TRAIN_FULL_MODEL   = False                                             # set True only in the LAST notebook
UNDERSAMPLE_NEG    = 0.5                                               # 1.0=full, 0.5=keep 50%% of negatives (~1.7x faster, no val/test impact)

# OPTIONAL: pre-built token cache from the "tokenize once" notebook.
# Set to "" to disable (tokenize from scratch -- takes ~8 min).
# Set to the path produced by your tokenize-once notebook commit, e.g.:
#   "/kaggle/input/<your-username>/dede-tokenize-once/stage1/outputs/token_cache"
EXTERNAL_TOKEN_CACHE = ""
# ▲▲▲ END EDIT ▲▲▲

# Install peft (Kaggle base image has transformers but not peft).
# Kaggle pre-installs torchao 0.10 which conflicts with peft>=0.19's strict
# version check (peft expects torchao > 0.16). We don't use torchao at all,
# so the safest fix is to uninstall it.
!pip uninstall -y torchao 2>&1 | tail -2
!pip install -q peft==0.19.1 2>&1 | tail -3

# Copy code to /kaggle/working/stage1 (writeable). /kaggle/input/ is read-only.
WORK_DIR = Path("/kaggle/working/stage1")
WORK_DIR.mkdir(parents=True, exist_ok=True)
for f in Path(CODE_DIR).glob("*.py"):
    shutil.copy(f, WORK_DIR / f.name)
print("Copied to:", WORK_DIR)
print("Files:", sorted(p.name for p in WORK_DIR.glob("*.py")))

os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))
"""


# ═════════════════════════════════════════════════════════════════════════
# CELL 3 -- Set env vars so the subprocess (CELL 5) finds paths correctly
# ═════════════════════════════════════════════════════════════════════════
"""
import os
# These env vars are read by config.py -- they override the local-dev defaults.
# Subprocesses inherit os.environ, so train.py will see them.
os.environ["STAGE1_PROFILE"]       = "kaggle"
os.environ["STAGE1_POSTS_CLEAN"]   = f"{PREP_DATA_DIR}/posts_clean.parquet"
os.environ["STAGE1_SPLITS"]        = f"{PREP_DATA_DIR}/splits.parquet"
os.environ["STAGE1_OUT_DIR"]       = "/kaggle/working/stage1/outputs"

# Verify
for k in ("STAGE1_PROFILE", "STAGE1_POSTS_CLEAN", "STAGE1_SPLITS", "STAGE1_OUT_DIR"):
    print(f"{k:25s} = {os.environ[k]}")
    if "POSTS_CLEAN" in k or "SPLITS" in k:
        print(f"{'':25s}   exists = {os.path.exists(os.environ[k])}")

# Verify GPU
import torch
print(f"\\nCUDA: {torch.cuda.is_available()}  n_gpus={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
"""


# ═════════════════════════════════════════════════════════════════════════
# CELL 4 -- Get token cache (from external dataset or build fresh)
# ═════════════════════════════════════════════════════════════════════════
"""
import shutil, subprocess
from pathlib import Path

LOCAL_CACHE = Path("/kaggle/working/stage1/outputs/token_cache")
LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)

if EXTERNAL_TOKEN_CACHE and Path(EXTERNAL_TOKEN_CACHE).exists():
    # Fast path: copy pre-built cache from the tokenize-once dataset
    print(f"Found external token cache: {EXTERNAL_TOKEN_CACHE}")
    if LOCAL_CACHE.exists():
        shutil.rmtree(LOCAL_CACHE)
    # Copy the whole token_cache directory (it contains tokenizer subfolders)
    print("Copying to local working dir ...")
    !cp -r "{EXTERNAL_TOKEN_CACHE}" "{LOCAL_CACHE.parent}/"
    result = subprocess.run(["du", "-sh", str(LOCAL_CACHE)],
                            capture_output=True, text=True)
    print(f"Cache ready ({result.stdout.strip()})")
else:
    if EXTERNAL_TOKEN_CACHE:
        print(f"WARNING: EXTERNAL_TOKEN_CACHE set to '{EXTERNAL_TOKEN_CACHE}' "
              "but does not exist. Falling back to fresh tokenization.")
    else:
        print("No EXTERNAL_TOKEN_CACHE set; tokenizing from scratch (~8 min).")

# Either way, build_or_load_tokenized will detect existing cache and load it.
from utils import load_tokenizer
from data   import build_or_load_tokenized

tok, tok_name = load_tokenizer(MODEL_INPUT_DIR)
ds = build_or_load_tokenized(tok, tok_name, force_rebuild=False)
print(f"\\nDataset ready: {len(ds):,} rows, columns={ds.column_names}")

import numpy as np
lens = np.array([len(x) for x in ds[:1000]['input_ids']])
print(f"Seq length p50/p95/p99 on first 1000: {np.percentile(lens, [50,95,99]).astype(int).tolist()}")
"""


# ═════════════════════════════════════════════════════════════════════════
# CELL 5 -- Train ONE fold (~3-4 hours on 2x T4)
# ═════════════════════════════════════════════════════════════════════════
"""
import subprocess, time

if TRAIN_FULL_MODEL:
    cmd = ["python", "train.py", "--mode", "full",
           "--model_dir", MODEL_INPUT_DIR,
           "--undersample_neg", str(UNDERSAMPLE_NEG)]
    label = "full"
else:
    cmd = ["python", "train.py", "--mode", "fold",
           "--fold", str(FOLD_THIS_NOTEBOOK),
           "--model_dir", MODEL_INPUT_DIR,
           "--undersample_neg", str(UNDERSAMPLE_NEG)]
    label = f"fold {FOLD_THIS_NOTEBOOK}"

print(f"=== Training {label} ===")
print("CMD:", " ".join(cmd))
t0 = time.perf_counter()

# Stream stdout in real-time so we can watch progress in the notebook
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
proc.wait()

elapsed_min = (time.perf_counter() - t0) / 60
print(f"\\n=== Done in {elapsed_min:.1f} min  (exit code {proc.returncode}) ===")
if proc.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {proc.returncode}")
"""


# ═════════════════════════════════════════════════════════════════════════
# CELL 6 -- (OPTIONAL) Resume from a saved checkpoint after a timeout
# ═════════════════════════════════════════════════════════════════════════
"""
# If your previous notebook session timed out mid-training, the latest
# checkpoint should still exist under outputs/checkpoints/{fold_K | full}/.
# Re-run this notebook with the same FOLD setting and CELL 5 below:

import subprocess, time

if TRAIN_FULL_MODEL:
    cmd = ["python", "train.py", "--mode", "full",
           "--model_dir", MODEL_INPUT_DIR, "--resume", "auto",
           "--undersample_neg", str(UNDERSAMPLE_NEG)]
else:
    cmd = ["python", "train.py", "--mode", "fold",
           "--fold", str(FOLD_THIS_NOTEBOOK),
           "--model_dir", MODEL_INPUT_DIR, "--resume", "auto",
           "--undersample_neg", str(UNDERSAMPLE_NEG)]

print("CMD:", " ".join(cmd))
t0 = time.perf_counter()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print(f"\\nResumed run done in {(time.perf_counter()-t0)/60:.1f} min")
"""


# ═════════════════════════════════════════════════════════════════════════
# CELL 7 -- Zip predictions for download (small) + summary
# ═════════════════════════════════════════════════════════════════════════
"""
import subprocess
from pathlib import Path

PRED_DIR = Path("/kaggle/working/stage1/outputs/predictions")
print("Files in predictions/:")
for p in sorted(PRED_DIR.glob("*.parquet")):
    print(f"  {p.name:50s} {p.stat().st_size/1e6:.1f} MB")

# Zip just predictions (small). Skip checkpoints (large, ~500 MB per fold).
zip_path = "/kaggle/working/predictions.zip"
subprocess.run(["zip", "-jrq", zip_path, str(PRED_DIR)], check=True)
print(f"\\nWrote {zip_path}  "
      f"({Path(zip_path).stat().st_size/1e6:.1f} MB)")
print("Download from the right sidebar -> Output panel.")

# Optionally, save the best checkpoint of THIS notebook's run for re-use:
# subprocess.run(["zip", "-rq", "/kaggle/working/checkpoint.zip",
#                 "outputs/checkpoints/"], check=True)
"""
