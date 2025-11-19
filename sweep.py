#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep --L and --kl_threshold for the greedy speculative decoding pipeline.

Features:
- Full checkpointing: resumes from where it left off.
- Skips combinations that already produced CSV results.
- Logs all activity to sweep_results_kl/sweep_<timestamp>.log
"""

import subprocess
import os
import json
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
L_values = [4, 8, 12, 16]
kl_thresholds = [-1.0, 1.0, 2.0, 3.0, 4.0]
dataset = "hardcoded"
draft_model = "meta-llama/Llama-2-7b-chat-hf"
target_model = "meta-llama/Llama-2-70b-chat-hf"
cache_dir = "/home1/10899/kimopro/SCRATCH/ml_data"
base_output_dir = "./sweep_results_kl"
max_new_tokens = 120
verbosity = 1  # number of -v flags

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs(base_output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(base_output_dir, f"sweep_{timestamp}.log")
checkpoint_path = os.path.join(base_output_dir, "checkpoint.json")

print(f"🚀 Starting parameter sweep with checkpointing. Log: {logfile}")

# -----------------------------------------------------------------------------
# Load checkpoint
# -----------------------------------------------------------------------------
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        completed = set(json.load(f))
    print(f"🔁 Loaded checkpoint: {len(completed)} runs already completed.")
else:
    completed = set()

# -----------------------------------------------------------------------------
# Generate parameter grid
# -----------------------------------------------------------------------------
param_grid = [(L, kl) for kl in kl_thresholds for L in L_values]

# -----------------------------------------------------------------------------
# Sweep loop
# -----------------------------------------------------------------------------
with open(logfile, "a") as log:
    for L, kl in param_grid:
        run_id = f"L{L}_KL{str(kl).replace('.', '_')}"
        csv_path = os.path.join(base_output_dir, f"results_{run_id}.csv")
        telemetry_path = os.path.join(base_output_dir, f"telemetry_{run_id}.jsonl")

        # Skip already finished
        if run_id in completed or os.path.exists(csv_path):
            print(f"✅ Skipping {run_id} (already done).")
            continue

        print(f"--> Launching run: L={L}, KL_Threshold={kl}")
        log.write(f"\n=== Running {run_id} ===\n")

        cmd = [
            "python3", "pipeline.py",
            "--dataset", dataset,
            "--cache_dir", cache_dir,
            "--draft_model", draft_model,
            "--target_model", target_model,
            "--max_new_tokens", str(max_new_tokens),
            "--L", str(L),
            "--kl_threshold", str(kl),
            "--csv_output", csv_path,
            "--output", telemetry_path,
            "-" + "v" * verbosity,
        ]

        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        try:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.PIPE, text=True)
            for stderr_line in iter(process.stderr.readline, ""):
                log.write(stderr_line)
                log.flush()

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            log.write(f"--- Run {run_id} finished successfully. ---\n")
            log.flush()

            # ✅ Update checkpoint
            completed.add(run_id)
            with open(checkpoint_path, "w") as f:
                json.dump(sorted(list(completed)), f, indent=2)

        except subprocess.CalledProcessError as e:
            error_message = f"❌ Run failed for {run_id}: {e}\n"
            print(error_message)
            log.write(error_message)
            log.flush()
            # Keep checkpoint intact so failed runs can retry later
            continue

print(f"\n🏁 Sweep complete. Logs and CSVs in: {base_output_dir}")
print(f"Checkpoint file: {checkpoint_path}")
