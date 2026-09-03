-- ============================================================
-- StrataCredit | Gold Layer
-- 04_loss_severity.sql
--
-- Loss severity analysis for terminated credit-event loans.
-- Only includes loans with qualifying zero balance codes.
-- ============================================================

CREATE OR REPLACE TABLE gold.loss_severity AS
SELECT
    loan_id,
    zero_balance_date,
    terminal_event_type,
    origination_year,
    fico,
    original_ltv,
    original_cltv,
    dti,
    property_state,
    loan_age,
    mortgage_insurance_pct,
    occupancy,
    loan_purpose,

    -- Balance at termination
    zero_balance_removal_upb                                    AS termination_upb,

    -- Recovery components
    net_sale_proceeds,
    mi_recoveries,
    non_mi_recoveries,
    expenses,
    delinquent_accrued_interest,
    actual_loss,

    -- Computed severity: realized loss / termination UPB
    CASE
        WHEN zero_balance_removal_upb IS NOT NULL
         AND zero_balance_removal_upb > 0
        THEN actual_loss / zero_balance_removal_upb
        ELSE NULL
    END                                                         AS loss_severity,

    -- Total recoveries
    COALESCE(mi_recoveries, 0)
        + COALESCE(non_mi_recoveries, 0)                        AS total_recoveries,

    -- Recovery rate
    CASE
        WHEN zero_balance_removal_upb IS NOT NULL
         AND zero_balance_removal_upb > 0
        THEN (COALESCE(mi_recoveries, 0) + COALESCE(non_mi_recoveries, 0))
             / zero_balance_removal_upb
        ELSE NULL
    END                                                         AS recovery_rate

FROM silver.loan_month_panel
WHERE is_credit_event = TRUE
  AND zero_balance_removal_upb IS NOT NULL
  AND actual_loss IS NOT NULL
ORDER BY zero_balance_date, loan_id
;
