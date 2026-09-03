# Methodology

## Prepayment Analytics

### SMM (Single Monthly Mortality)

For each active, non-delinquent loan in reporting month t:

```
scheduled_principal_t = UPB_{t-1} × (r/12) / ((1 + r/12)^n_{t-1} − 1)

where:
  r   = prior month interest rate (annual %)
  n   = prior remaining term (months)

SMM_t = max(0, (UPB_{t-1} − UPB_t − scheduled_principal_t) / UPB_{t-1})
```

For loans that voluntarily prepaid in month t: `SMM_t = 1.0`

Severely delinquent loans (60+ DPD) are excluded from the SMM calculation
because their balance changes reflect servicer advances, not voluntary prepayment.

### CPR (Conditional Prepayment Rate)

```
CPR_t = 1 − (1 − SMM_t)^12
```

Expressed as a percentage (0–100).

Pool-level CPR uses UPB-weighted average SMM:

```
WA_SMM_t = Σ(SMM_i × UPB_i) / Σ(UPB_i)
CPR_pool  = 1 − (1 − WA_SMM_t)^12
```

## Credit Events

### Definitions

**Serious Delinquency**: Loan reaches 90+ days past due (delinquency_months ≥ 3).

**Terminal Credit Event**: Loan terminates via a non-voluntary zero balance code:
- `02` — Third-party sale
- `03` — Short sale / charge-off
- `09` — REO disposition
- `15` — Note sale

**Voluntary Prepayment**: Zero balance code `01`.

**Other Terminations**: Repurchases (`06`, `96`, `97`), reperforming sales (`16`).

These concepts are kept separate. Not every delinquent loan is a realized default.

### CDR (Conditional Default Rate)

StrataCredit's internal CDR definition:

```
monthly_rate_t = new_credit_events_t / active_loans_{t−1}

CDR_t = 1 − (1 − monthly_rate_t)^12
```

**Important**: This is StrataCredit's internal methodology. It does NOT
replicate Fitch, Moody's, KBRA, S&P, or any other rating-agency CDR definition.

## Loss Severity

```
loss_severity = actual_loss / zero_balance_removal_upb
```

Where:
- `actual_loss` = `actual_loss_calculation` from Freddie Mac servicer data
- `zero_balance_removal_upb` = UPB at time of termination

Only computed for terminal credit-event loans with both fields populated.

Negative severities (over-recoveries where MI + sale proceeds exceed UPB)
are retained and flagged rather than silently removed.

### Recovery Components

```
total_recoveries = mi_recoveries + non_mi_recoveries
recovery_rate    = total_recoveries / zero_balance_removal_upb
```

## Weighted Averages

All portfolio-level weighted averages use current UPB as the weight:

```
WA_metric = Σ(metric_i × UPB_i) / Σ(UPB_i)
```

Null values in the metric column are excluded from both numerator and denominator.

## Stress Scenarios

Stress multipliers are applied to base model predictions:

```
stressed_CE_rate  = base_CE_rate  × default_multiplier
stressed_severity = base_severity × severity_multiplier
stressed_prepay   = base_CPR      × prepayment_multiplier

annual_gross_loss = pool_UPB × stressed_CE_rate × stressed_severity
```

These are deterministic sensitivity tools. They do not represent
rating-agency models or stress methodologies.

## Temporal Validation

Model evaluation uses origination-vintage splits:

| Split      | Vintages  |
|------------|-----------|
| Train      | 2012–2017 |
| Validation | 2018–2019 |
| Test       | 2020–2022 |

Random train/test splits are not used as the primary evaluation because
they introduce temporal leakage: future loan performance information can
contaminate training if rows from the same loan appear in both splits.
