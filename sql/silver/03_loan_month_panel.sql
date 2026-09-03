-- ============================================================
-- StrataCredit | Silver Layer
-- 03_loan_month_panel.sql
--
-- CANONICAL analytical dataset.
-- One row = one loan × one reporting month.
--
-- Combines origination characteristics (static) with
-- performance state as of that reporting month (dynamic).
-- This is the primary input for all analytics, modeling,
-- and Gold table construction.
-- ============================================================

CREATE OR REPLACE TABLE silver.loan_month_panel AS
SELECT
    -- ── Identity ──────────────────────────────────────────────────────
    p.loan_id,
    p.reporting_period,

    -- ── Origination Characteristics (static) ─────────────────────────
    o.origination_date,
    o.first_payment_date,
    o.maturity_date,
    o.original_upb,
    o.original_interest_rate,
    o.original_loan_term,
    o.original_ltv,
    o.original_cltv,
    o.fico,
    o.dti,
    o.mortgage_insurance_pct,
    o.num_units,
    o.num_borrowers,
    o.loan_purpose,
    o.occupancy,
    o.property_type,
    o.property_state,
    o.channel,
    o.msa,
    o.first_time_homebuyer,
    o.interest_only,
    o.super_conforming,
    o.prepayment_penalty,
    o.amort_type,
    YEAR(o.origination_date)    AS origination_year,
    QUARTER(o.origination_date) AS origination_quarter,

    -- ── Performance State as of Reporting Period ──────────────────────
    p.current_upb,
    p.deferred_upb,
    p.interest_bearing_upb,
    p.current_interest_rate,
    p.loan_age,
    p.remaining_term,
    p.delinquency_months,
    p.delinquency_state,
    p.raw_delinquency_status,
    p.modification_flag,
    p.step_modification_flag,
    p.deferred_payment_plan,
    p.borrower_assistance_status_code,
    p.estimated_ltv,
    p.zero_balance_code,
    p.zero_balance_date,
    p.zero_balance_removal_upb,

    -- ── Loss Components ───────────────────────────────────────────────
    p.net_sale_proceeds,
    p.mi_recoveries,
    p.non_mi_recoveries,
    p.expenses,
    p.legal_costs,
    p.maintenance_costs,
    p.taxes_insurance,
    p.miscellaneous_expenses,
    p.actual_loss,
    p.delinquent_accrued_interest,

    -- ── Derived Fields ────────────────────────────────────────────────
    YEAR(p.reporting_period)    AS reporting_year,
    MONTH(p.reporting_period)   AS reporting_month,

    -- UPB ratio (current / original) — proxy for scheduled vs actual
    CASE WHEN o.original_upb > 0
        THEN p.current_upb / o.original_upb
        ELSE NULL
    END                         AS upb_ratio,

    -- Terminal event classification
    CASE zero_balance_code
        WHEN '01' THEN 'voluntary_prepayment'
        WHEN '02' THEN 'third_party_sale'
        WHEN '03' THEN 'short_sale'
        WHEN '06' THEN 'repurchase'
        WHEN '09' THEN 'reo_disposition'
        WHEN '15' THEN 'note_sale'
        WHEN '16' THEN 'reperforming_sale'
        ELSE CASE WHEN zero_balance_code IS NOT NULL
            THEN 'other_terminal'
            ELSE NULL
        END
    END                         AS terminal_event_type,

    -- Whether this row is a terminal event row
    CASE WHEN zero_balance_code IS NOT NULL
        AND zero_balance_code <> ''
        THEN TRUE ELSE FALSE
    END                         AS is_terminal,

    -- Whether this is a voluntary prepayment
    CASE WHEN zero_balance_code = '01' THEN TRUE ELSE FALSE END
                                AS is_voluntary_prepayment,

    -- Whether this is a credit event (non-voluntary termination)
    CASE WHEN zero_balance_code IN ('02','03','09','15')
        THEN TRUE ELSE FALSE END
                                AS is_credit_event,

    -- Serious delinquency (90+)
    CASE WHEN delinquency_months >= 3
        OR delinquency_state IN ('DQ90PLUS','REO','TERMINAL')
        THEN TRUE ELSE FALSE END
                                AS is_seriously_delinquent,

    p.delinquency_due_to_disaster,

    -- ── Source Metadata ───────────────────────────────────────────────
    p._source_vintage           AS vintage

FROM silver.loan_performance p
INNER JOIN silver.loan_origination o
    ON p.loan_id = o.loan_id
;

-- Create a summary statistics view for quick inspection
CREATE OR REPLACE VIEW silver.panel_summary AS
SELECT
    COUNT(DISTINCT loan_id)                                     AS unique_loans,
    COUNT(*)                                                    AS total_rows,
    MIN(reporting_period)                                       AS earliest_period,
    MAX(reporting_period)                                       AS latest_period,
    MIN(origination_year)                                       AS earliest_vintage,
    MAX(origination_year)                                       AS latest_vintage,
    SUM(CASE WHEN is_voluntary_prepayment THEN 1 ELSE 0 END)    AS voluntary_prepayments,
    SUM(CASE WHEN is_credit_event         THEN 1 ELSE 0 END)    AS credit_events,
    SUM(CASE WHEN is_seriously_delinquent THEN 1 ELSE 0 END)    AS serious_delinquencies,
    ROUND(SUM(actual_loss), 0)                                  AS total_realized_losses
FROM silver.loan_month_panel
;
