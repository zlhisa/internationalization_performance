# Leverage and Financial Performance of European SMEs
### Research Note Project | SS 2026

## Author
Lisa [Nachname]

## Research Question
Does leverage negatively affect the financial performance of 
European SMEs?

## Hypotheses
- H1: Leverage is negatively associated with firm performance (ROA).

## Theoretical Background
Higher leverage increases financial risk and interest obligations, 
which can reduce profitability. For SMEs in particular, excessive 
debt may constrain investment and operational flexibility, leading 
to lower financial performance. This is consistent with the 
trade-off theory of capital structure, which suggests that beyond 
an optimal debt level, the costs of financial distress outweigh 
the tax benefits of debt (Modigliani & Miller, 1963).

## Data
| Item | Detail |
|------|--------|
| Source | WRDS / Compustat Global |
| Table | comp_global_daily.g_funda |
| Fiscal years | 2015-2024 |
| Downloaded | 2026-06-09 |
| Raw rows | [aus pull_metadata.txt] |
| Clean rows | [aus clean_log.txt] |

## Empirical Strategy
The hypothesis will be tested using a simple linear regression 
model in Python:
ROA = beta0 + beta1 * Leverage + controls + error

## Variables

### Dependent variable (Y)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| ROA | ni, at | ni / at |

### Independent variable (X)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Leverage | dltt, dlc, seq | (dltt + dlc) / seq |

### Controls
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Firm Size | at | log(at) |
| Cash Flow | ibc, dp, at | (ibc + dp) / at |

## Project Structure
The project follows a reproducible research setup with separate 
folders for raw data, processed data, code, output, references 
and the final research note.