import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

csv_path = Path(
    "/home/yentl/pytorch_gammanet/outputs_jitter_contour_bias_final/"
    "csv/jitter_h1_contour_enrichment_summary.csv"
)

out_path = csv_path.parent / "baseline_vs_matched_bias_metrics_table.tex"

# ==========================================================
# Load data
# ==========================================================

df = pd.read_csv(csv_path)

# Keep only baseline and matched-bias rows
baseline = df[df["condition"] == "baseline"].copy()

matched = df[
    ((df["contour_type"] == "C") & (df["condition"] == "C_bias")) |
    ((df["contour_type"] == "straight") & (df["condition"] == "straight_bias"))
].copy()

baseline["Condition"] = "Baseline"
matched["Condition"] = "Bias"

table_df = pd.concat([baseline, matched], ignore_index=True)

# Nice contour labels
table_df["Contour"] = table_df["contour_type"].replace({
    "C": "C",
    "straight": "Straight"
})

# Sort: C first, then Straight; within each jitter baseline then bias
contour_order = {"C": 0, "Straight": 1}
condition_order = {"Baseline": 0, "Bias": 1}

table_df["contour_order"] = table_df["Contour"].map(contour_order)
table_df["condition_order"] = table_df["Condition"].map(condition_order)

table_df = table_df.sort_values(
    ["contour_order", "jitter", "condition_order"]
).reset_index(drop=True)

# ==========================================================
# Generate LaTeX longtable
# ==========================================================

latex = []

latex.append(r"\begin{longtable}{lllcccc}")
latex.append(
    r"\caption{\textbf{Comparison of contour-selectivity metrics between baseline "
    r"and matched top-down modulation in $h1_{exc}$.}}"
)
latex.append(r"\label{tab:baseline_bias_metrics}\\")
latex.append(r"\toprule")
latex.append(
    r"\textbf{Contour} & "
    r"\textbf{Condition} & "
    r"\textbf{Jitter} & "
    r"\textbf{Preference ratio} & "
    r"\textbf{Activation (\%)} & "
    r"\textbf{Enrichment} & "
    r"\textbf{Dice overlap} \\"
)
latex.append(r"\midrule")
latex.append(r"\endfirsthead")

latex.append(r"\toprule")
latex.append(
    r"\textbf{Contour} & "
    r"\textbf{Condition} & "
    r"\textbf{Jitter} & "
    r"\textbf{Preference ratio} & "
    r"\textbf{Activation (\%)} & "
    r"\textbf{Enrichment} & "
    r"\textbf{Dice overlap} \\"
)
latex.append(r"\midrule")
latex.append(r"\endhead")

last_contour = None

for _, row in table_df.iterrows():
    if last_contour is not None and row["Contour"] != last_contour:
        latex.append(r"\midrule")

    latex.append(
        f"{row['Contour']} & "
        f"{row['Condition']} & "
        f"J{int(row['jitter']):02d} & "
        f"{row['mean_contour_preference_ratio']:.3f} & "
        f"{row['mean_activation_on_contour_pct']:.3f} & "
        f"{row['mean_contour_enrichment']:.3f} & "
        f"{row['mean_dice_top_activation']:.3f} \\\\"
    )

    last_contour = row["Contour"]

latex.append(r"\bottomrule")
latex.append(r"\end{longtable}")

latex_code = "\n".join(latex)

# ==========================================================
# Save
# ==========================================================

out_path.write_text(latex_code)

print(f"Saved LaTeX table to:\n{out_path}")