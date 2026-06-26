# Leverage and Financial Performance of European SMEs
### Research Note Project | SS 2026

## Author
Lisa [Nachname]

## Research Question
Does leverage negatively affect the financial performance of 
European SMEs, and does firm size moderate this relationship?

## Hypotheses
- H1: Leverage is negatively associated with firm performance (ROA).
- H2: Firm size negatively moderates the leverage-ROA relationship 
(larger firms are more resilient to debt).

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
| Raw rows | 338,483 |
| Clean rows | 26,083 |

## Empirical Strategy
The hypothesis will be tested using a panel regression model in Python:
ROA = beta0 + beta1 * Leverage + beta2 * Size + beta3 * 
Leverage x Size + controls + error

## Variables
| Variable | Field(s) | Formula | Role |
|----------|----------|---------|------|
| ROA | ib, at | ib / at | Dependent (Y) |
| Leverage | dltt, at | dltt / at | Independent (X) |
| Lev x Size | - | leverage x ln_at | H2 interaction |
| Firm size | at | log(at) | Moderator + Control |
| CAPX intensity | capx, at | capx / at | Control |
| Cash ratio | che, at | che / at | Control |

## Project Structure
The project follows a reproducible research setup with separate 
folders for raw data, processed data, code, output, references 
and the final research note.