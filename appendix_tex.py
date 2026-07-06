import pandas as pd

# ==========================================================
# Load channel classification
# ==========================================================

csv_path = "/home/yentl/pytorch_gammanet/outputs_bias_contour_final/plots/05_channel_classification/channel_classification/h1_channel_classes_from_mean_enrichment.csv"

df = pd.read_csv(csv_path)

# sort channels
df = df.sort_values("channel").reset_index(drop=True)

# channels 1-128 instead of 0-127
df["channel"] = df["channel"] + 1

# nicer labels
df["channel_class"] = (
    df["channel_class"]
    .replace({
        "C": "C",
        "straight": "Straight"
    })
)

# round numbers
for col in ["C", "straight", "C_minus_straight_enrichment"]:
    df[col] = df[col].round(3)

# ==========================================================
# Generate LaTeX table
# ==========================================================

latex = []

latex.append(r"\begin{longtable}{ccccc}")
latex.append(r"\caption{Contour-selective classification of all feature channels in $h1_{exc}$.}")
latex.append(r"\label{tab:all_channel_classes}\\")
latex.append(r"\toprule")
latex.append(r"\textbf{Channel} & \textbf{C enrichment} & \textbf{Straight enrichment} & \textbf{$\Delta$ enrichment} & \textbf{Classification} \\")
latex.append(r"\midrule")
latex.append(r"\endfirsthead")

latex.append(r"\toprule")
latex.append(r"\textbf{Channel} & \textbf{C enrichment} & \textbf{Straight enrichment} & \textbf{$\Delta$ enrichment} & \textbf{Classification} \\")
latex.append(r"\midrule")
latex.append(r"\endhead")

for _, row in df.iterrows():

    latex.append(
        f"{int(row['channel'])} & "
        f"{row['C']:.3f} & "
        f"{row['straight']:.3f} & "
        f"{row['C_minus_straight_enrichment']:.3f} & "
        f"{row['channel_class']} \\\\"
    )

latex.append(r"\bottomrule")
latex.append(r"\end{longtable}")

latex_code = "\n".join(latex)

# ==========================================================
# Save
# ==========================================================

outfile = "/home/yentl/pytorch_gammanet/outputs_bias_contour_final/plots/05_channel_classification/channel_classification/appendix_channel_table.tex"

with open(outfile, "w") as f:
    f.write(latex_code)

print(f"Saved to:\n{outfile}")