-- ============================================================
-- StrataCredit | Gold Layer
-- 03_delinquency_roll_rates.sql
--
-- Monthly delinquency transition matrix.
-- Calculates roll-forward and roll-back rates between states.
-- States: CURRENT, DQ30, DQ60, DQ90PLUS, TERMINAL
-- ============================================================

CREATE OR REPLACE TABLE gold.delinquency_roll_rates AS
WITH transitions AS (
    SELECT
        loan_id,
        reporting_period,
        origination_year,
        fico,
        original_ltv,
        dti,
        property_state,
        loan_age,
        delinquency_state                                       AS state_t,
        LEAD(delinquency_state) OVER (
            PARTITION BY loan_id
            ORDER BY reporting_period
        )                                                       AS state_t1,
        current_upb
    FROM silver.loan_month_panel
    WHERE delinquency_state NOT IN ('UNKNOWN')
)
SELECT
    DATE_TRUNC('month', reporting_period)                       AS cohort_month,
    origination_year,
    state_t                                                     AS from_state,
    state_t1                                                    AS to_state,
    COUNT(*)                                                    AS loan_count,
    SUM(current_upb)                                            AS upb,
    -- Transition rate: count of (from->to) / count of (from)
    100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (
        PARTITION BY DATE_TRUNC('month', reporting_period),
                     origination_year,
                     state_t
    ), 0)                                                       AS transition_rate_pct
FROM transitions
WHERE state_t  IS NOT NULL
  AND state_t1 IS NOT NULL
GROUP BY 1, 2, 3, 4
ORDER BY cohort_month, from_state, to_state
;

-- Convenience view: standard transition matrix (Current Month)
CREATE OR REPLACE VIEW gold.transition_matrix_latest AS
SELECT *
FROM gold.delinquency_roll_rates
WHERE cohort_month = (
    SELECT MAX(cohort_month) FROM gold.delinquency_roll_rates
)
;
