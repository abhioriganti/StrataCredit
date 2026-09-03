-- ============================================================
-- StrataCredit | Gold Layer
-- 01_portfolio_snapshot.sql
--
-- Weighted-average portfolio metrics for a given cut-off date.
-- Parameterised by :cutoff_date (pass via DuckDB prepared stmt).
-- ============================================================

CREATE OR REPLACE TABLE gold.portfolio_snapshot AS
SELECT
    MAX(reporting_period)                                       AS as_of_date,
    COUNT(DISTINCT loan_id)                                     AS loan_count,
    SUM(current_upb)                                            AS total_current_upb,
    AVG(current_upb)                                            AS avg_loan_balance,

    -- Weighted averages (UPB-weighted)
    SUM(current_interest_rate * current_upb)
        / NULLIF(SUM(current_upb), 0)                           AS wa_coupon,
    SUM(fico * current_upb)
        / NULLIF(SUM(CASE WHEN fico IS NOT NULL THEN current_upb ELSE 0 END), 0)
                                                                AS wa_fico,
    SUM(original_ltv * current_upb)
        / NULLIF(SUM(CASE WHEN original_ltv IS NOT NULL THEN current_upb ELSE 0 END), 0)
                                                                AS wa_ltv,
    SUM(original_cltv * current_upb)
        / NULLIF(SUM(CASE WHEN original_cltv IS NOT NULL THEN current_upb ELSE 0 END), 0)
                                                                AS wa_cltv,
    SUM(dti * current_upb)
        / NULLIF(SUM(CASE WHEN dti IS NOT NULL THEN current_upb ELSE 0 END), 0)
                                                                AS wa_dti,
    SUM(loan_age * current_upb)
        / NULLIF(SUM(current_upb), 0)                           AS wala,
    SUM(remaining_term * current_upb)
        / NULLIF(SUM(current_upb), 0)                           AS warm,

    -- Purpose mix (% of UPB)
    100.0 * SUM(CASE WHEN loan_purpose = 'P' THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS purchase_pct,
    100.0 * SUM(CASE WHEN loan_purpose IN ('C','R') THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS refinance_pct,

    -- Occupancy mix
    100.0 * SUM(CASE WHEN occupancy = 'P' THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS owner_occupied_pct,
    100.0 * SUM(CASE WHEN occupancy = 'I' THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS investor_pct,

    -- MI share
    100.0 * SUM(CASE WHEN mortgage_insurance_pct > 0 THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS mi_share_pct,

    -- Delinquency breakdown
    100.0 * SUM(CASE WHEN delinquency_state = 'CURRENT' THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS current_pct,
    100.0 * SUM(CASE WHEN delinquency_state = 'DQ30'    THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS dq30_pct,
    100.0 * SUM(CASE WHEN delinquency_state = 'DQ60'    THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS dq60_pct,
    100.0 * SUM(CASE WHEN delinquency_state IN ('DQ90PLUS','REO') THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS dq90plus_pct,

    -- Concentration risk
    100.0 * SUM(CASE WHEN fico < 660           THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS low_fico_pct,
    100.0 * SUM(CASE WHEN original_ltv > 80    THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS high_ltv_pct,
    100.0 * SUM(CASE WHEN modification_flag    THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS modified_pct

FROM silver.loan_month_panel
WHERE delinquency_state NOT IN ('TERMINAL')
  AND is_terminal = FALSE
;
