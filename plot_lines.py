import os
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "sweep_results_kl"   # folder containing all results_L*_KL*.csv

# Regex that matches:
#  - KL=-1, 1, 2, 3...
#  - KL=1_5, 2_5, 3_5 (representing 1.5, 2.5, 3.5)
#  - Works with or without "_0"
pattern = re.compile(
    r"results_L(?P<L>\d+)_KL(?P<KL>-?\d+(?:_\d+)?)"
)

records = []

# Walk all files
for f in os.listdir(ROOT):
    match = pattern.match(f)
    if match:
        L = int(match.group("L"))

        # Convert KL notation: "1_5" → "1.5"
        KL_raw = match.group("KL")
        KL = float(KL_raw.replace("_", "."))

        path = os.path.join(ROOT, f)

        df = pd.read_csv(path)

        mean_speedup = df["speedup_vs_target"].mean()
        mean_prefix = df["accepted_mean_prefix_len"].mean()

        records.append({
            "L": L,
            "KL": KL,
            "mean_speedup": mean_speedup,
            "mean_prefix": mean_prefix,
        })

# Create dataframe of aggregated results
agg = pd.DataFrame(records)
agg = agg.sort_values(["L", "KL"])

print(agg)

# Create a big figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Plot 1: Speedup vs target
for L, group in agg.groupby("L"):
    axes[0].plot(group["KL"], group["mean_speedup"], marker="o", label=f"L={L}")

axes[0].set_title("Mean Speedup vs Target")
axes[0].set_ylabel("Mean speedup_vs_target")
axes[0].grid(True)
axes[0].legend(title="L")

# Plot 2: Accepted mean prefix length
for L, group in agg.groupby("L"):
    axes[1].plot(group["KL"], group["mean_prefix"], marker="o", label=f"L={L}")

axes[1].set_title("Mean Accepted Prefix Length")
axes[1].set_xlabel("KL")
axes[1].set_ylabel("Mean accepted_mean_prefix_len")
axes[1].grid(True)
axes[1].legend(title="L")

plt.tight_layout()
plt.savefig("aggregated_results.png")
print("Saved plot to aggregated_results.png")
