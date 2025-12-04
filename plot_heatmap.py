import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = "sweep_results_kl"

# Regex supports KL = 1, -1, 1_5, etc.
pattern = re.compile(r"results_L(?P<L>\d+)_KL(?P<KL>[\d_.-]+)\.csv")

records = []

# Parse Data
if not os.path.exists(ROOT):
    print(f"Directory {ROOT} not found.")
else:
    for f in os.listdir(ROOT):
        match = pattern.search(f)
        if match:
            L = int(match.group("L"))
            kl_str = match.group("KL").replace('_', '.')

            try:
                KL = float(kl_str)
            except ValueError:
                continue

            path = os.path.join(ROOT, f)
            try:
                df = pd.read_csv(path)
                if "speedup_vs_target" in df.columns:
                    mean_speedup = df["speedup_vs_target"].mean()
                    records.append({
                        "L": L,
                        "KL": KL,
                        "mean_speedup": mean_speedup,
                    })
            except Exception as e:
                print(f"Error reading {f}: {e}")

agg = pd.DataFrame(records)

if agg.empty:
    print("No data found.")
else:
    agg = agg.sort_values(["L", "KL"])

    # Heatmap
    pivot_table = agg.pivot(index="KL", columns="L", values="mean_speedup")
    pivot_table = pivot_table.sort_index(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Speedup (x)'})
    plt.title("Speedup Heatmap")
    plt.xlabel("L")
    plt.ylabel("KL Threshold")
    plt.tight_layout()
    plt.savefig("aggregated_results_heatmap.png")
    print("Heatmap saved.")

    # Generate Minimal LaTeX Table

    latex_rows = []

    for L_val in sorted(agg["L"].unique()):
        subset = agg[agg["L"] == L_val]

        # Baseline KL=-1
        baseline_row = subset[subset["KL"] == -1.0]
        if baseline_row.empty:
            baseline_speedup = float("nan")
            baseline_str = "N/A"
        else:
            baseline_speedup = baseline_row.iloc[0]["mean_speedup"]
            baseline_str = f"{baseline_speedup:.2f}"

        # Best KL
        best_idx = subset["mean_speedup"].idxmax()
        best_row = subset.loc[best_idx]
        best_kl = best_row["KL"]
        best_speedup = best_row["mean_speedup"]

        # % Improvement
        if pd.notna(baseline_speedup) and baseline_speedup > 0:
            pct_change = ((best_speedup - baseline_speedup) / baseline_speedup) * 100
            pct_str = f"+{pct_change:.1f}\\%" if pct_change >= 0 else f"{pct_change:.1f}\\%"
        else:
            pct_str = "N/A"

        latex_rows.append(
            f"{L_val} & {baseline_str} & {best_kl} & {best_speedup:.2f} & {pct_str} \\\\"
        )

    # FULL MINIMAL LATEX DOCUMENT
    latex_table = (
        "\\documentclass{article}\n\n"
        "\\begin{document}\n\n"
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\begin{tabular}{c c c c c}\n"
        "\\hline\n"
        "L & Baseline Speedup (KL = -1) & Best KL & Best Speedup & Improvement vs Baseline \\\\\n"
        "\\hline\n"
        + "\n".join(latex_rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Comparison of best KL threshold vs no threshold (KL = -1) across different lookahead steps (L).}\n"
        "\\label{tab:kl_improvement}\n"
        "\\end{table}\n\n"
        "\\end{document}\n"
    )

    with open("summary_table.tex", "w") as f:
        f.write(latex_table)

    print("\n--- Generated Minimal LaTeX Table ---")
    print(latex_table)
    print("-----------------------------")
    print("LaTeX table saved to 'summary_table.tex'")
