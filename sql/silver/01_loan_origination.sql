-- ============================================================
-- StrataCredit | Silver Layer
-- 01_loan_origination.sql
--
-- One record per mortgage loan.
-- Combines all origination-vintage Parquet shards into a
-- standardised loan origination table.
-- ============================================================

CREATE OR REPLACE TABLE silver.loan_origination AS
SELECT
    -- ── Primary Key ───────────────────────────────────────────────────
    loan_sequence_number                       AS loan_id,

    -- ── Dates ─────────────────────────────────────────────────────────
    CASE WHEN SUBSTR(first_payment_date, 1, 4) BETWEEN '1900' AND '2100'
         THEN STRPTIME(first_payment_date, '%Y%m')
         ELSE STRPTIME(first_payment_date, '%m%Y') END
                                               AS first_payment_date,
    -- Origination date approximated from first payment (1 month prior)
    DATE_TRUNC('month',
        (CASE WHEN SUBSTR(first_payment_date, 1, 4) BETWEEN '1900' AND '2100'
              THEN STRPTIME(first_payment_date, '%Y%m')
              ELSE STRPTIME(first_payment_date, '%m%Y') END) - INTERVAL '1 month'
    )                                          AS origination_date,
    CASE WHEN SUBSTR(maturity_date, 1, 4) BETWEEN '1900' AND '2100'
         THEN STRPTIME(maturity_date, '%Y%m')
         ELSE STRPTIME(maturity_date, '%m%Y') END
                                               AS maturity_date,

    -- ── Loan Characteristics ──────────────────────────────────────────
    original_upb,
    original_interest_rate,
    original_loan_term,
    ltv                                        AS original_ltv,
    cltv                                       AS original_cltv,
    credit_score                               AS fico,
    dti,
    mi_pct                                     AS mortgage_insurance_pct,
    num_units,
    num_borrowers,

    -- ── Categorical Attributes ────────────────────────────────────────
    loan_purpose,
    occupancy_status                           AS occupancy,
    property_type,
    property_state,
    channel,
    postal_code,
    msa,

    -- ── Flags ─────────────────────────────────────────────────────────
    CASE WHEN first_time_homebuyer_flag = 'Y' THEN TRUE ELSE FALSE END
                                               AS first_time_homebuyer,
    CASE WHEN interest_only_indicator   = 'Y' THEN TRUE ELSE FALSE END
                                               AS interest_only,
    CASE WHEN super_conforming_flag     = 'Y' THEN TRUE ELSE FALSE END
                                               AS super_conforming,
    CASE WHEN prepayment_penalty_flag   = 'Y' THEN TRUE ELSE FALSE END
                                               AS prepayment_penalty,
    amort_type,
    program_indicator,

    -- ── Source Metadata ───────────────────────────────────────────────
    _source_file,
    _source_vintage,
    _ingested_at,
    _file_checksum

FROM read_parquet(
    'data/bronze/origination/**/*.parquet',
    hive_partitioning = TRUE
)
;

-- Index equivalent: create unique constraint check
-- (DuckDB does not enforce PK but we validate in the quality module)
