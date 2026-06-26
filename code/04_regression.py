"""
04_regression.py
----------------
Panel fixed-effects regressions testing H1 and H2.

Input:  data/processed/panel_with_vars.parquet
Output: output/tables/regression_results.csv

Research design
---------------
Y:   RoA            = ib / at
X:   R&D intensity  = xrd / at
Mod: Firm size      = log(at)
Int: rd_x_size      = rd_intensity * ln_at

Hypotheses
----------
H1: β(rd_intensity) < 0 and significant
    R&D intensity negatively affects RoA in the short run due to immediate
    expensing under IFRS. Performance returns accrue with a 2-5 year lag.
    (Test: β significant, negative direction expected)

H2: β(rd_x_size) > 0
    Firm size positively moderates the R&D intensity-RoA relationship.
    Larger SMEs have greater absorptive capacity and can spread fixed R&D
    costs over a larger asset base.
    (Test: β(rd_intensity x ln_at) > 0)

Models
------
(1) Pooled OLS     — baseline, ignores panel structure
(2) TWFE           — two-way FE (firm + year), main specification
(3) TWFE + H2      — adds rd_x_size interaction term

OLS vs FE
---------
Large difference expected — unobserved firm heterogeneity (management
quality, innovation culture) correlates with R&D investment, causing
substantial omitted variable bias in pooled OLS.

Usage
-----
    python code/04_regression.py
    task regression
"""

import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects

warnings.filterwarnings("ignore")


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

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_PATH    = Path("data") / "processed" / "panel_with_vars.parquet"
TABLE_PATH = Path("output") / "tables"
TABLE_PATH.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("\nLoading panel with variables...")
df = pd.read_parquet(IN_PATH)
print(f"  Shape: {df.shape[0]:,} rows | {df['gvkey'].nunique():,} firms | "
      f"years {df['fyear'].min()}-{df['fyear'].max()}")
print(f"  High leverage firms: {(df['leverage']>df['leverage'].median()).sum():,} "
      f"({(df['leverage']>df['leverage'].median()).mean()*100:.1f}%)")

# ── Variables ─────────────────────────────────────────────────────────────────
# ADAPT: change these to your own variable names
DV       = "roa"
X_MAIN   = "leverage"
INTERACT = "lev_x_size"
CONTROLS = ["ln_at", "capx_intensity", "cash_ratio"]

# Build interaction if not already present
if "lev_x_size" not in df.columns:
    df["lev_x_size"] = df["leverage"] * df["ln_at"]

# ── Regression sample ─────────────────────────────────────────────────────────
reg_vars = [DV, X_MAIN, INTERACT] + CONTROLS
df_reg   = df.dropna(subset=reg_vars).copy()
df_panel = df_reg.set_index(["gvkey", "fyear"])
print(f"\nRegression sample: {len(df_reg):,} obs | "
      f"{df_reg['gvkey'].nunique():,} firms")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_se(res, var):
    """Handles statsmodels (bse) and linearmodels (std_errors)."""
    if hasattr(res, "std_errors"):
        return res.std_errors[var]   # linearmodels
    return res.bse[var]              # statsmodels

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

# ── Model 1: Pooled OLS ───────────────────────────────────────────────────────
print("\n── Model 1: Pooled OLS ──────────────────────────────")
f1   = f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)}"
res1 = smf.ols(f1, data=df_reg).fit(cov_type="HC3")
print(f"  N={int(res1.nobs):,} | R²={res1.rsquared:.3f} | "
      f"{X_MAIN}: {res1.params[X_MAIN]:.4f}{stars(res1.pvalues[X_MAIN])}")

# ── Model 2: TWFE (main) ──────────────────────────────────────────────────────
print("── Model 2: Two-Way Fixed Effects ───────────────────")
f2   = (f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)} "
        f"+ EntityEffects + TimeEffects")
res2 = PanelOLS.from_formula(f2, data=df_panel).fit(
    cov_type="clustered", cluster_entity=True)
print(f"  N={int(res2.nobs):,} | R²(within)={res2.rsquared:.3f} | "
      f"{X_MAIN}: {res2.params[X_MAIN]:.4f}{stars(res2.pvalues[X_MAIN])}")

# ── Model 3: TWFE + H2 interaction ───────────────────────────────────────────
print("── Model 3: TWFE + H2 Interaction ───────────────────")
f3   = (f"{DV} ~ {X_MAIN} + {INTERACT} + {' + '.join(CONTROLS)} "
        f"+ EntityEffects + TimeEffects")
res3 = PanelOLS.from_formula(f3, data=df_panel).fit(
    cov_type="clustered", cluster_entity=True)
b_int = res3.params.get(INTERACT, np.nan)
p_int = res3.pvalues.get(INTERACT, 1)
print(f"  N={int(res3.nobs):,} | R²(within)={res3.rsquared:.3f} | "
      f"{INTERACT}: {b_int:.4f}{stars(p_int)}")

# ── Random Effects + Hausman comparison ───────────────────────────────────────
print("── Random Effects (Hausman comparison) ──────────────")
f_re   = f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)}"
res_re = RandomEffects.from_formula(f_re, data=df_panel).fit()
diff   = abs(res2.params[X_MAIN] - res_re.params[X_MAIN])
print(f"  FE: {res2.params[X_MAIN]:.4f} | "
      f"RE: {res_re.params[X_MAIN]:.4f} | diff: {diff:.4f}")
print(f"  {'Prefer FE (non-trivial difference)' if diff > 0.005 else 'Small difference — FE still recommended in IB research'}")

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("Regression Results")
print("="*65)

all_vars = [X_MAIN, INTERACT] + CONTROLS
models   = [res1, res2, res3]
labels   = ["(1) OLS", "(2) TWFE", "(3) TWFE+H2"]

rows = []
for var in all_vars:
    row = {"Variable": var}
    for label, res in zip(labels, models):
        if var in res.params.index:
            b  = res.params[var]
            se = get_se(res, var)
            p  = res.pvalues[var]
            row[label]         = f"{b:.4f}{stars(p)}"
            row[f"{label}_se"] = f"({se:.4f})"
        else:
            row[label]         = ""
            row[f"{label}_se"] = ""
    rows.append(row)

stats_rows = [
    {"Variable": "Firm FE",
     "(1) OLS": "No",  "(2) TWFE": "Yes", "(3) TWFE+H2": "Yes"},
    {"Variable": "Year FE",
     "(1) OLS": "No",  "(2) TWFE": "Yes", "(3) TWFE+H2": "Yes"},
    {"Variable": "Clustered SE",
     "(1) OLS": "No",  "(2) TWFE": "Yes", "(3) TWFE+H2": "Yes"},
    {"Variable": "N",
     "(1) OLS":     f"{int(res1.nobs):,}",
     "(2) TWFE":    f"{int(res2.nobs):,}",
     "(3) TWFE+H2": f"{int(res3.nobs):,}"},
    {"Variable": "R²",
     "(1) OLS":     f"{res1.rsquared:.3f}",
     "(2) TWFE":    f"{res2.rsquared:.3f}",
     "(3) TWFE+H2": f"{res3.rsquared:.3f}"},
]
rows.extend(stats_rows)

results_df = pd.DataFrame(rows).set_index("Variable")
print(results_df[labels].to_string())
print("\n* p<0.10  ** p<0.05  *** p<0.01")
print("SEs in parentheses. Models (2)-(3): clustered at firm level.")

results_df.to_csv(TABLE_PATH / "regression_results.csv")
print(f"\nSaved regression_results.csv")

# ── H1 Diagnostic ─────────────────────────────────────────────────────────────
print("\n── H1 Diagnostic ────────────────────────────────────")
b_x = res2.params[X_MAIN]
p_x = res2.pvalues[X_MAIN]
print(f"  β(leverage) = {b_x:.4f}{stars(p_x)} (p={p_x:.3f})")
if p_x < 0.10:
    if b_x < 0:
        print("  H1 SUPPORTED: negative effect as predicted")
        print("  Higher leverage increases financial risk and reduces RoA")
        print("  Consistent with trade-off theory of capital structure")
    else:
        print("  H1 NOT SUPPORTED: effect is positive (unexpected direction)")
        print("  Check: winsorizing, sample selection, variable construction")
else:
    print(f"  H1 NOT SUPPORTED: not significant (p={p_x:.3f})")
    print("  Consider: too few R&D firms for within-firm identification")

# ── H2 Diagnostic ─────────────────────────────────────────────────────────────
print("\n── H2 Diagnostic ────────────────────────────────────")
print(f"  β(lev_x_size) = {b_int:.4f}{stars(p_int)} (p={p_int:.3f})")
if p_int < 0.10:
    if b_int > 0:
        print("  H2 SUPPORTED: larger SMEs benefit more from R&D investment")
        print("  Consistent with absorptive capacity theory")
    else:
        print("  H2 NOT SUPPORTED in expected direction: negative moderation")
        print("  Smaller firms may benefit more from R&D at the margin")
else:
    print(f"  H2 NOT SUPPORTED: interaction not significant (p={p_int:.3f})")

# ── OLS vs TWFE ───────────────────────────────────────────────────────────────
print("\n── OLS vs TWFE comparison ───────────────────────────")
ols_b    = res1.params[X_MAIN]
fe_b     = res2.params[X_MAIN]
pct_diff = abs((ols_b - fe_b) / ols_b) * 100
print(f"  OLS β = {ols_b:.4f} | FE β = {fe_b:.4f} | difference = {pct_diff:.1f}%")
print(f"  R² OLS = {res1.rsquared:.3f} | R² within FE = {res2.rsquared:.3f}")
if pct_diff > 20:
    print("  Large difference → substantial omitted variable bias in OLS")
    print("  Firm FE absorbs time-invariant heterogeneity correlated with R&D")

print(f"\nDone. Results in output/tables/regression_results.csv")
