"""
03_descriptives.py
------------------
Variable construction, winsorizing, summary statistics and figures.

Input:  data/processed/panel_clean.parquet
Output: data/processed/panel_with_vars.parquet
        output/tables/summary_statistics.csv
        output/figures/correlation_matrix.png
        output/figures/dv_distribution.png
        output/figures/main_relationship.png

Research design
---------------
Y:   RoA            = ib / at
X:   R&D intensity  = xrd / at  (missing xrd -> 0)
Mod: Firm size      = log(at)
Int: rd_x_size      = rd_intensity * ln_at  (H2)
Controls:
     leverage       = dltt / at
     capx_intensity = capx / at
     cash_ratio     = che / at

Note on DOI
-----------
pifo (foreign income) is not available in this Compustat Global pull.
rect/sale produces extreme outliers for SMEs with low/volatile sales.
The research question was therefore updated to focus on R&D intensity
as the main independent variable — well-grounded in absorptive capacity
theory (Cohen & Levinthal 1990) and available with 100% coverage.

Usage
-----
    python code/03_descriptives.py
    task descriptives
"""

import os, sys, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Find project root ─────────────────────────────────────────────────────────
def find_env():
    current = Path(os.getcwd())
    for path in [current] + list(current.parents):
        if (path / ".env").exists():
            return path / ".env"
        try:
            for s in path.iterdir():
                if s.is_dir() and (s / ".env").exists():
                    return s / ".env"
        except PermissionError:
            continue
    raise FileNotFoundError("Could not find .env anywhere.")

project_root = find_env().parent
os.chdir(project_root)
print(f"Project root: {project_root}")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif"})
WU_BLUE = "#002f5f"
WU_RED  = "#c8102e"

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_PATH    = Path("data") / "processed" / "panel_clean.parquet"
OUT_PANEL  = Path("data") / "processed" / "panel_with_vars.parquet"
TABLE_PATH = Path("output") / "tables"
FIG_PATH   = Path("output") / "figures"
TABLE_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("\nLoading clean panel...")
df = pd.read_parquet(IN_PATH)
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Data quality filters ──────────────────────────────────────────────────────
print("\nApplying data quality filters...")
n = len(df)
df = df[(df["at"] > 0.1) & (df["sale"] > 0) & (df["seq"] > 0)].copy()
print(f"  After at>0.1, sale>0, seq>0: {len(df):,} (removed {n-len(df):,})")

# ── SME filter ────────────────────────────────────────────────────────────────
n = len(df)
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
df = df[sme_mask].copy()
print(f"  After SME filter (emp<250 OR at<=43m): {len(df):,} (removed {n-len(df):,})")

# ── Variable construction ──────────────────────────────────────────────────────
print("\nConstructing variables...")

# Dependent variable
df["roa"] = df["ib"] / df["at"]

# Independent variable: R&D intensity
df["rd_intensity"] = df["xrd"].fillna(0) / df["at"]

# Moderator + control: firm size
df["ln_at"] = df["at"].apply(lambda x: math.log(x) if x > 0 else np.nan)

# H2 interaction
df["rd_x_size"] = df["rd_intensity"] * df["ln_at"]

# Controls
df["leverage"]       = df["dltt"].fillna(0) / df["at"]
df["capx_intensity"] = df["capx"].fillna(0) / df["at"]
df["cash_ratio"]     = df["che"].fillna(0)  / df["at"]

# ── Drop missing core variables ───────────────────────────────────────────────
CORE_VARS = ["roa", "rd_intensity", "ln_at", "leverage"]
n = len(df)
df = df.dropna(subset=CORE_VARS).copy()
print(f"  Dropped {n-len(df):,} rows with missing core vars")
print(f"  Working sample: {len(df):,} firm-years | {df['gvkey'].nunique():,} firms")

# ── Winsorize at 1%-99% ───────────────────────────────────────────────────────
def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

print("\nWinsorizing at 1%-99%...")
for col in ["roa", "rd_intensity", "leverage", "capx_intensity", "cash_ratio"]:
    df[col] = winsorize(df[col])
    print(f"  {col:<20} [{df[col].min():>8.4f}, {df[col].max():>8.4f}]")

# Recompute interaction after winsorizing
df["rd_x_size"] = df["rd_intensity"] * df["ln_at"]

# ── Minimum 3 observations per firm ──────────────────────────────────────────
obs   = df.groupby("gvkey")["fyear"].count()
valid = obs[obs >= 3].index
n = len(df)
df = df[df["gvkey"].isin(valid)].copy()
print(f"\nMin 3 obs: {n:,} -> {len(df):,} | {df['gvkey'].nunique():,} firms")
print(f"R&D firms (rd>0): {(df['rd_intensity']>0).sum():,} "
      f"({(df['rd_intensity']>0).mean()*100:.1f}%)")

# ── Summary statistics ────────────────────────────────────────────────────────
VAR_LABELS = {
    "roa":            "RoA (ib/at)",
    "rd_intensity":   "R&D Intensity (xrd/at)",
    "ln_at":          "Firm Size (log assets)",
    "leverage":       "Leverage (dltt/at)",
    "capx_intensity": "CAPX Intensity (capx/at)",
    "cash_ratio":     "Cash Ratio (che/at)",
}

summary = (
    df[list(VAR_LABELS.keys())]
    .rename(columns=VAR_LABELS)
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T[["count","mean","std","min","25%","50%","75%","max"]]
    .round(4)
)
print("\n=== Summary Statistics ===")
print(summary.to_string())
summary.to_csv(TABLE_PATH / "summary_statistics.csv")
print(f"\nSaved summary_statistics.csv")

# ── Correlation matrix ────────────────────────────────────────────────────────
corr = df[list(VAR_LABELS.keys())].rename(columns=VAR_LABELS).corr().round(2)
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Research Variables",
             fontsize=13, color=WU_BLUE)
fig.tight_layout()
fig.savefig(FIG_PATH / "correlation_matrix.png", dpi=150)
plt.close()
print("Saved correlation_matrix.png")

# ── DV distribution + median RoA by year ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["roa"], bins=60, color=WU_BLUE, alpha=0.8, edgecolor="white")
axes[0].axvline(df["roa"].mean(),   color=WU_RED,   lw=2,
                label=f"Mean   = {df['roa'].mean():.3f}")
axes[0].axvline(df["roa"].median(), color="orange", lw=2, ls="--",
                label=f"Median = {df['roa'].median():.3f}")
axes[0].set_xlabel("RoA")
axes[0].set_title("Distribution of RoA", color=WU_BLUE)
axes[0].legend()

yearly = df.groupby("fyear")["roa"].median()
axes[1].bar(yearly.index, yearly.values, color=WU_BLUE, alpha=0.8)
axes[1].axhline(0, color="black", lw=0.8, ls="--")
axes[1].set_xlabel("Fiscal Year")
axes[1].set_ylabel("Median RoA")
axes[1].set_title("Median RoA by Year", color=WU_BLUE)
fig.tight_layout()
fig.savefig(FIG_PATH / "dv_distribution.png", dpi=150)
plt.close()
print("Saved dv_distribution.png")

# ── Main relationship: R&D intensity vs RoA ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
df_plot = df.reset_index(drop=True)

# Left: scatter + bin means for R&D firms only
df_rd = df_plot[df_plot["rd_intensity"] > 0].reset_index(drop=True)
axes[0].scatter(df_rd["rd_intensity"], df_rd["roa"],
                alpha=0.1, s=8, color=WU_BLUE, label="Has R&D")
bins = pd.cut(df_rd["rd_intensity"], bins=15)
bm   = df_rd.groupby(bins, observed=True)[["rd_intensity","roa"]].mean()
axes[0].plot(bm["rd_intensity"], bm["roa"],
             color=WU_RED, lw=2.5, label="Bin mean")
axes[0].axhline(0, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("R&D Intensity (xrd/at)")
axes[0].set_ylabel("RoA")
axes[0].set_title("R&D Intensity vs. RoA\n(firms with R&D > 0 only)", color=WU_BLUE)
axes[0].legend()

# Right: median RoA by firm size bin — No R&D vs Has R&D
df_plot["rd_group"]  = np.where(df_plot["rd_intensity"] == 0,
                                "No R&D (xrd=0)", "Has R&D (xrd>0)")
df_plot["size_bin"]  = pd.cut(df_plot["ln_at"], bins=10)
palette2 = {"No R&D (xrd=0)": "#2166ac", "Has R&D (xrd>0)": WU_RED}
for label, group in df_plot.groupby("rd_group", observed=True):
    g  = group.reset_index(drop=True)
    bm = g.groupby("size_bin", observed=True)[["ln_at","roa"]].median()
    axes[1].plot(bm["ln_at"], bm["roa"], lw=2,
                 label=label, color=palette2[label],
                 marker="o", markersize=5)
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("Firm Size (log assets)")
axes[1].set_ylabel("Median RoA")
axes[1].set_title("Median RoA by Firm Size:\nNo R&D vs Has R&D",
                  color=WU_BLUE)
axes[1].legend()

fig.suptitle("Main Relationship: R&D Intensity -> RoA",
             fontsize=13, color=WU_BLUE, y=1.02)
fig.tight_layout()
fig.savefig(FIG_PATH / "main_relationship.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved main_relationship.png")

# ── Save panel with variables ─────────────────────────────────────────────────
df.to_parquet(OUT_PANEL, index=False)
print(f"\nSaved panel_with_vars.parquet: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("Next step: python code/04_regression.py")