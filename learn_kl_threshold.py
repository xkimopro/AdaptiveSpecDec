#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learns an optimal KL divergence threshold (v2 - Corrected Logic).

This version correctly simulates the "predictive fallback" strategy by modeling
the replacement of a speculative window with a single autoregressive step
when the previous window's KL divergence exceeds a threshold.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_telemetry_files(log_directory: str) -> List[Dict[str, float]]:
    """
    Parses all telemetry_*.jsonl files in a directory to extract
    window data into a sequential list.
    """
    jsonl_files = glob.glob(os.path.join(log_directory, "telemetry_*.jsonl"))
    if not jsonl_files:
        print(f"Error: No telemetry_*.jsonl files found in '{log_directory}'")
        return []

    print(f"Found {len(jsonl_files)} telemetry files to analyze.")
    windows_data = []
    # Process files in a consistent order to maintain sequence integrity
    for file_path in sorted(jsonl_files):
        try:
            with open(file_path, "r") as f:
                kl_mean_buffer = None
                for line in f:
                    try:
                        data = json.loads(line)
                        if "window_kl_mean" in data:
                            kl_mean_buffer = data["window_kl_mean"]
                        elif "window" in data and kl_mean_buffer not in [None, "sample_finished"]:
                            accepted_len = data["window"]["accepted_prefix_length"]
                            windows_data.append(
                                {"kl": kl_mean_buffer, "accepted": accepted_len}
                            )
                            kl_mean_buffer = None
                        elif "sample_finished" in data:
                            # Add a marker to reset the state for each sample
                            windows_data.append({"kl": -1, "accepted": -1, "marker": "sample_finished"})
                    except (json.JSONDecodeError, TypeError):
                        continue
        except IOError as e:
            print(f"Warning: Could not read file {file_path}. Error: {e}")

    return [w for w in windows_data if "marker" not in w]


def simulate_performance(
    windows_data: List[Dict[str, float]], threshold: float
) -> float:
    """
    Simulates the performance of the predictive fallback strategy.

    The performance metric is "Simulated Speedup", calculated as the
    total number of tokens generated per target model forward pass.
    """
    total_tokens_generated = 0
    total_target_passes = len(windows_data) # Each window is one pass
    
    if total_target_passes == 0:
        return 0.0

    in_fallback_mode = False

    for i, window in enumerate(windows_data):
        # Decide action for the CURRENT step based on the PREVIOUS step's state
        if in_fallback_mode:
            # This step is a forced autoregressive step.
            # It generates 1 token (the "extra" target token).
            total_tokens_generated += 1
            # After the fallback, we always return to speculation.
            in_fallback_mode = False
        else:
            # This is a normal speculative step.
            # The number of tokens generated is what was observed in the log.
            # This includes the "bonus" token.
            total_tokens_generated += window["accepted"] + 1

        # Now, look at the current window's KL to decide the fate of the NEXT step.
        if window["kl"] > threshold:
            in_fallback_mode = True

    return total_tokens_generated / total_target_passes


def find_optimal_threshold(
    windows_data: List[Dict[str, float]],
) -> Tuple[float, float]:
    """Searches for the KL threshold that maximizes the simulated speedup."""
    # A more reasonable search space for mean KL
    search_space = np.arange(0.5, 3.5, 0.05)
    best_threshold = -1.0
    max_speedup = 0.0

    print(f"\nSearching for optimal threshold across {len(search_space)} candidates...")
    
    results = []
    for threshold in search_space:
        speedup = simulate_performance(windows_data, threshold)
        results.append((threshold, speedup))
        if speedup > max_speedup:
            max_speedup = speedup
            best_threshold = threshold
    
    print("\n--- Top 5 Thresholds ---")
    top_5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    for i, (thr, spd) in enumerate(top_5):
        print(f"{i+1}. Threshold: {thr:.2f} -> Simulated Speedup: {spd:.4f} Tokens/Pass")

    return best_threshold, max_speedup


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Learn an optimal KL threshold from telemetry logs (v2)."
    )
    parser.add_argument(
        "log_directory",
        type=str,
        default=".",
        nargs="?",
        help="The directory containing the telemetry_*.jsonl files (default: current directory).",
    )
    args = parser.parse_args()

    windows_data = parse_telemetry_files(args.log_directory)
    if not windows_data:
        print("No valid window data could be parsed. Exiting.")
        return

    print(f"\nSuccessfully parsed {len(windows_data)} speculative windows from all files.")
    best_threshold, max_speedup = find_optimal_threshold(windows_data)

    # --- Baseline Calculation ---
    baseline_tokens = sum(w['accepted'] + 1 for w in windows_data)
    baseline_speedup = baseline_tokens / len(windows_data) if windows_data else 0

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(f"Baseline Performance (No Controller): {baseline_speedup:.4f} Tokens/Pass")
    print(f"Optimal Performance (With Controller):  {max_speedup:.4f} Tokens/Pass")
    print("-" * 60)
    print(f"The optimal KL divergence threshold is: {best_threshold:.2f}")
    print("\nRecommendation:")
    print(f"Using '--kl_threshold {best_threshold:.2f}' in your benchmark script is")
    print("predicted to provide the best performance trade-off.")
    print("=" * 60)


if __name__ == "__main__":
    main()