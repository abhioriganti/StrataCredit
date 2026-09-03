-- ============================================================
-- StrataCredit | Silver Layer
-- 02_loan_performance.sql
--
-- One record per loan per reporting month.
-- Standardises Freddie Mac performance files into a clean
-- analytical schema with typed fields and null-safe sentinels.
-- ============================================================

CREATE OR REPLACE TABLE silver.loan_performance AS
SELECT
    -- ── Primary Key (composite) ───────────────────────────────────────
    loan_sequence_number                                        AS loan_id,
    CASE WHEN SUBSTR(monthly_reporting_period, 1, 4) BETWEEN '1900' AND '2100'
         THEN STRPTIME(monthly_reporting_period, '%Y%m')
         ELSE STRPTIME(monthly_reporting_period, '%m%Y') END    AS reporting_period,

    -- ── Balance ───────────────────────────────────────────────────────
    current_actual_upb                                          AS current_upb,
    COALESCE(current_deferred_upb, 0.0)                         AS deferred_upb,
    COALESCE(interest_bearing_upb, current_actual_upb)          AS interest_bearing_upb,
    zero_balance_removal_upb,

    -- ── Rate & Terms ──────────────────────────────────────────────────
    current_interest_rate,
    loan_age,
    remaining_months_to_legal_maturity                          AS remaining_term,

    -- ── Delinquency ───────────────────────────────────────────────────
    current_loan_delinquency_status                             AS raw_delinquency_status,
    -- Normalise to integer months delinquent (0 = current)
    CASE
        WHEN current_loan_delinquency_status IN ('0','00')  THEN 0
        WHEN current_loan_delinquency_status = 'RA'         THEN 99  -- REO
        WHEN TRY_CAST(current_loan_delinquency_status AS INTEGER) IS NOT NULL
            THEN CAST(current_loan_delinquency_status AS INTEGER)
        ELSE NULL
    END                                                         AS delinquency_months,

    -- ── Delinquency State Buckets ─────────────────────────────────────
    CASE
        WHEN current_loan_delinquency_status IN ('0','00')          THEN 'CURRENT'
        WHEN current_loan_delinquency_status = '1'                  THEN 'DQ30'
        WHEN current_loan_delinquency_status = '2'                  THEN 'DQ60'
        WHEN TRY_CAST(current_loan_delinquency_status AS INTEGER) >= 3 THEN 'DQ90PLUS'
        WHEN current_loan_delinquency_status = 'RA'                 THEN 'REO'
        WHEN zero_balance_code IS NOT NULL AND zero_balance_code <> ''
                                                                    THEN 'TERMINAL'
        ELSE 'UNKNOWN'
    END                                                         AS delinquency_state,

    -- ── Modifications ─────────────────────────────────────────────────
    CASE WHEN modification_flag = 'Y' THEN TRUE ELSE FALSE END  AS modification_flag,
    CASE WHEN step_modification_flag = 'Y' THEN TRUE ELSE FALSE END
                                                                AS step_modification_flag,
    CASE WHEN deferred_payment_plan = 'Y' THEN TRUE ELSE FALSE END
                                                                AS deferred_payment_plan,
    borrower_assistance_status_code,

    -- ── Terminal Events ───────────────────────────────────────────────
    zero_balance_code,
    CASE WHEN SUBSTR(zero_balance_effective_date, 1, 4) BETWEEN '1900' AND '2100'
         THEN TRY_STRPTIME(zero_balance_effective_date, '%Y%m')
         ELSE TRY_STRPTIME(zero_balance_effective_date, '%m%Y') END
                                                                AS zero_balance_date,

    -- ── Estimated LTV ─────────────────────────────────────────────────
    estimated_ltv,

    -- ── Loss Components ───────────────────────────────────────────────
    CASE
        WHEN net_sale_proceeds = 'C' THEN NULL   -- 'C' = credit to borrower
        ELSE TRY_CAST(net_sale_proceeds AS DOUBLE)
    END                                                         AS net_sale_proceeds,
    mi_recoveries,
    non_mi_recoveries,
    expenses,
    legal_costs,
    maintenance_costs,
    taxes_insurance,
    miscellaneous_expenses,
    actual_loss_calculation                                     AS actual_loss,
    modification_cost,
    delinquent_accrued_interest,

    -- ── Disaster Flag ─────────────────────────────────────────────────
    CASE WHEN delinquency_due_to_disaster = 'Y' THEN TRUE ELSE FALSE END
                                                                AS delinquency_due_to_disaster,

    -- ── Source Metadata ───────────────────────────────────────────────
    _source_file,
    _source_vintage,
    _ingested_at,
    _file_checksum

FROM read_parquet(
    'data/bronze/performance/**/*.parquet',
    hive_partitioning = TRUE
)
;
