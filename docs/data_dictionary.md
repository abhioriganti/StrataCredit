# Data Dictionary

## Silver Layer

### `silver.loan_origination`

One record per mortgage loan.

| Column | Type | Description |
|--------|------|-------------|
| loan_id | Utf8 | Freddie Mac loan sequence number (primary key) |
| origination_date | Date | Estimated origination date (first_payment_date − 1 month) |
| first_payment_date | Date | Date of first scheduled payment |
| maturity_date | Date | Scheduled maturity date |
| original_upb | Float64 | Original unpaid principal balance ($) |
| original_interest_rate | Float64 | Note rate at origination (%) |
| original_loan_term | Int64 | Original term in months |
| original_ltv | Float64 | Origination LTV (%) |
| original_cltv | Float64 | Combined LTV at origination (%) |
| fico | Int64 | Borrower FICO score at origination |
| dti | Float64 | Debt-to-income ratio at origination (%) |
| mortgage_insurance_pct | Float64 | MI coverage percentage; null if no MI |
| loan_purpose | Utf8 | P=Purchase, C=Cash-out refi, R=Rate refi |
| occupancy | Utf8 | P=Primary, S=Second home, I=Investor |
| property_type | Utf8 | SF=Single family, CO=Condo, MH=Manufactured |
| property_state | Utf8 | 2-letter state abbreviation |
| channel | Utf8 | R=Retail, B=Broker, C=Correspondent |
| first_time_homebuyer | Boolean | First-time homebuyer flag |
| interest_only | Boolean | Interest-only loan flag |

### `silver.loan_performance`

One record per loan per reporting month.

| Column | Type | Description |
|--------|------|-------------|
| loan_id | Utf8 | Loan identifier (FK to loan_origination) |
| reporting_period | Date | Monthly reporting period (first of month) |
| current_upb | Float64 | Current unpaid principal balance ($) |
| current_interest_rate | Float64 | Current note rate (%) |
| loan_age | Int64 | Months since origination |
| remaining_term | Int64 | Months remaining to maturity |
| delinquency_months | Int64 | Months past due (0 = current) |
| delinquency_state | Utf8 | CURRENT/DQ30/DQ60/DQ90PLUS/REO/TERMINAL |
| modification_flag | Boolean | Loan has been modified |
| zero_balance_code | Utf8 | Terminal event code; null if active |
| zero_balance_date | Date | Date of termination |
| zero_balance_removal_upb | Float64 | UPB at termination |
| actual_loss | Float64 | Realized loss from Freddie Mac servicer data |

### `silver.loan_month_panel`

Canonical analytical dataset. One row = one loan × one reporting month.
Combines all origination and performance fields plus derived indicators:

| Column | Type | Description |
|--------|------|-------------|
| is_terminal | Boolean | True if this row is the terminal event row |
| is_voluntary_prepayment | Boolean | True if ZBC = '01' |
| is_credit_event | Boolean | True if ZBC in {02, 03, 09, 15} |
| is_seriously_delinquent | Boolean | True if delinquency_months ≥ 3 |
| terminal_event_type | Utf8 | Descriptive termination label |

## Derived Fields

### Estimated LTV
Updated monthly by Freddie Mac servicers. Reflects current property value
estimates rather than origination appraisal.

### Delinquency State
Mapped from `current_loan_delinquency_status`:
- `"0"` or `"00"` → CURRENT
- `"1"` → DQ30
- `"2"` → DQ60
- `"3"` through `"12"` → DQ90PLUS
- `"RA"` → REO
- Non-null `zero_balance_code` → TERMINAL
