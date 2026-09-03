-- ============================================================
-- StrataCredit | Gold Layer
-- 02_cpr_history.sql
--
-- Monthly CPR (Conditional Prepayment Rate) history.
--
-- CPR Methodology:
--   For each loan in month t:
--     scheduled_factor = (1 + rate/12)^remaining_term_start
--                      / ((1 + rate/12)^remaining_term_start - 1)
--     But we use a simpler balance-decline approach:
--
--   SMM_t = (UPB_{t-1} - UPB_t - scheduled_principal_t)
--           / UPB_{t-1}
--
--   CPR_t = 1 - (1 - SMM_t)^12
--
-- Scheduled principal is approximated per standard amortization.
-- See methodology.md for full derivation.
-- ============================================================

CREATE OR REPLACE TABLE gold.cpr_history AS
WITH monthly_balances AS (
    SELECT
        loan_id,
        reporting_period,
        origination_year,
        origination_quarter,
        loan_age,
        current_upb,
        current_interest_rate,
        remaining_term,
        original_ltv,
        fico,
        loan_purpose,
        property_state,
        occupancy,
        -- Prior month UPB via LAG
        LAG(current_upb) OVER (
            PARTITION BY loan_id
            ORDER BY reporting_period
        ) AS prior_upb,
        LAG(current_interest_rate) OVER (
            PARTITION BY loan_id
            ORDER BY reporting_period
        ) AS prior_rate,
        LAG(remaining_term) OVER (
            PARTITION BY loan_id
            ORDER BY reporting_period
        ) AS prior_remaining_term,
        is_terminal,
        is_voluntary_prepayment,
        delinquency_state
    FROM silver.loan_month_panel
    WHERE is_terminal = FALSE
      OR is_voluntary_prepayment = TRUE  -- include prepayment month
),
smm_calc AS (
    SELECT
        loan_id,
        reporting_period,
        origination_year,
        loan_age,
        current_upb,
        prior_upb,
        current_interest_rate,
        remaining_term,
        original_ltv,
        fico,
        loan_purpose,
        property_state,
        occupancy,
        is_voluntary_prepayment,
        -- Scheduled principal: standard amortization formula
        -- P_sched = prior_upb * r / ((1+r)^n - 1)
        -- where r = monthly rate, n = prior remaining term
        CASE
            WHEN prior_upb IS NOT NULL
             AND prior_upb > 0
             AND prior_rate IS NOT NULL
             AND prior_rate > 0
             AND prior_remaining_term IS NOT NULL
             AND prior_remaining_term > 0
            THEN
                prior_upb * (prior_rate / 1200.0) /
                (POWER(1 + prior_rate / 1200.0, prior_remaining_term) - 1.0)
            ELSE 0.0
        END AS scheduled_principal,
        -- SMM = (prior_upb - current_upb - scheduled_principal) / prior_upb
        CASE
            WHEN prior_upb IS NOT NULL AND prior_upb > 0
             AND NOT is_voluntary_prepayment  -- for prepaid loans, SMM = 1
            THEN
                GREATEST(
                    (
                        prior_upb
                        - COALESCE(current_upb, 0)
                        - CASE
                            WHEN prior_rate IS NOT NULL AND prior_rate > 0
                             AND prior_remaining_term IS NOT NULL AND prior_remaining_term > 0
                            THEN prior_upb * (prior_rate / 1200.0) /
                                 (POWER(1 + prior_rate / 1200.0, prior_remaining_term) - 1.0)
                            ELSE 0.0
                          END
                    ) / prior_upb,
                    0.0
                )
            WHEN is_voluntary_prepayment THEN 1.0  -- full prepayment
            ELSE NULL
        END AS smm
    FROM monthly_balances
    WHERE prior_upb IS NOT NULL
      AND delinquency_state NOT IN ('DQ60','DQ90PLUS','REO')  -- exclude severely DQ loans from prepay
),
monthly_cpr AS (
    SELECT
        reporting_period,
        origination_year,
        loan_age,
        original_ltv,
        fico,
        loan_purpose,
        property_state,
        occupancy,
        COUNT(loan_id)                                          AS loan_count,
        SUM(current_upb)                                        AS total_upb,
        AVG(smm)                                                AS avg_smm,
        -- CPR = 1 - (1 - SMM)^12
        CASE WHEN AVG(smm) BETWEEN 0 AND 1
            THEN (1.0 - POWER(1.0 - AVG(smm), 12)) * 100.0
            ELSE NULL
        END                                                     AS cpr,
        -- UPB-weighted SMM
        SUM(smm * current_upb) / NULLIF(SUM(current_upb), 0)   AS wa_smm
    FROM smm_calc
    WHERE smm IS NOT NULL
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
)
SELECT
    reporting_period,
    origination_year,
    loan_age,
    loan_count,
    total_upb,
    avg_smm,
    wa_smm,
    -- CPR from wa_smm
    CASE WHEN wa_smm BETWEEN 0 AND 1
        THEN (1.0 - POWER(1.0 - wa_smm, 12)) * 100.0
        ELSE NULL
    END AS cpr_wa,
    cpr AS cpr_simple
FROM monthly_cpr
ORDER BY reporting_period, origination_year, loan_age
;
