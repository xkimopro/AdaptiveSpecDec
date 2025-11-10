#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep --L and --temperature for speculative decoding.

Runs pipeline.py for all combinations of L ∈ {4, 12, 20}
and temperature ∈ {0.2, 0.4}, saving results to separate CSVs.

Usage:
    python3 sweep_L_temp.py
"""

import subprocess
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
L_values = [4, 12, 20]                   # sweep over these L values
temperatures = [0.2, 0.4]                # sweep over these temperatures
num_samples = 10                         # samples per run
dataset = "alpaca-mini"
draft_model = "meta-llama/Llama-2-7b-chat-hf"
target_model = "meta-llama/Llama-2-70b-chat-hf"
cache_dir = "/home1/10899/kimopro/SCRATCH/ml_data"
base_output_dir = "./sweep_results"
max_new_tokens = 80
top_p = 1.0
verbosity = 1  # -v flag count

# -----------------------------------------------------------------------------
# Run setup
# -----------------------------------------------------------------------------
os.makedirs(base_output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(base_output_dir, f"sweep_{timestamp}.log")

with open(logfile, "w") as log:
    for temp in temperatures:
        for L in L_values:
            csv_path = os.path.join(base_output_dir, f"results_L{L}_T{temp}.csv")
            telemetry_path = os.path.join(base_output_dir, f"telemetry_L{L}_T{temp}.jsonl")

            cmd = [
                "python3", "pipeline.py",
                "--dataset", dataset,
                "--num_samples", str(num_samples),
                "--cache_dir", cache_dir,
                "--draft_model", draft_model,
                "--target_model", target_model,
                "--max_new_tokens", str(max_new_tokens),
                "--L", str(L),
                "--temperature", str(temp),
                "--top_p", str(top_p),
                "--csv_output", csv_path,
                "--output", telemetry_path,
                "-" + "v" * verbosity,
            ]

            log.write(f"\n=== Running L={L}, T={temp} ===\n")
            log.write(" ".join(cmd) + "\n")

            try:
                subprocess.run(cmd, check=True, stdout=log, stderr=log)
            except subprocess.CalledProcessError as e:
                log.write(f"Run failed for L={L}, T={temp}: {e}\n")

print(f"\n✅ Sweep complete. Logs and CSVs saved to: {base_output_dir}")
