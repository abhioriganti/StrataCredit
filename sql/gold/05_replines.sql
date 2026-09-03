-- ============================================================
-- StrataCredit | Gold Layer
-- 05_replines.sql
--
-- Representative collateral lines (replines) aggregation.
-- Groups loans into cells by FICO × LTV × coupon × vintage × state_group.
-- ============================================================

CREATE OR REPLACE TABLE gold.replines AS
SELECT
    -- ── Grouping Dimensions ────────────────────────────────────────────
    CASE
        WHEN fico < 620         THEN '< 620'
        WHEN fico < 640         THEN '620-639'
        WHEN fico < 660         THEN '640-659'
        WHEN fico < 680         THEN '660-679'
        WHEN fico < 700         THEN '680-699'
        WHEN fico < 720         THEN '700-719'
        WHEN fico < 740         THEN '720-739'
        WHEN fico < 760         THEN '740-759'
        WHEN fico < 780         THEN '760-779'
        ELSE                         '780+'
    END                                                         AS fico_band,

    CASE
        WHEN original_ltv <= 60 THEN '<= 60%'
        WHEN original_ltv <= 65 THEN '60-65%'
        WHEN original_ltv <= 70 THEN '65-70%'
        WHEN original_ltv <= 75 THEN '70-75%'
        WHEN original_ltv <= 80 THEN '75-80%'
        WHEN original_ltv <= 85 THEN '80-85%'
        WHEN original_ltv <= 90 THEN '85-90%'
        WHEN original_ltv <= 95 THEN '90-95%'
        ELSE                         '95%+'
    END                                                         AS ltv_band,

    CASE
        WHEN original_interest_rate < 3.0  THEN '< 3.0%'
        WHEN original_interest_rate < 3.5  THEN '3.0-3.5%'
        WHEN original_interest_rate < 4.0  THEN '3.5-4.0%'
        WHEN original_interest_rate < 4.5  THEN '4.0-4.5%'
        WHEN original_interest_rate < 5.0  THEN '4.5-5.0%'
        WHEN original_interest_rate < 5.5  THEN '5.0-5.5%'
        WHEN original_interest_rate < 6.0  THEN '5.5-6.0%'
        WHEN original_interest_rate < 7.0  THEN '6.0-7.0%'
        ELSE                                    '7.0%+'
    END                                                         AS coupon_band,

    origination_year                                            AS vintage,

    CASE
        WHEN property_state IN ('CT','MA','ME','NH','NJ','NY','PA','RI','VT') THEN 'Northeast'
        WHEN property_state IN ('AL','AR','FL','GA','KY','LA','MS','NC','SC','TN','VA','WV') THEN 'Southeast'
        WHEN property_state IN ('IA','IL','IN','KS','MI','MN','MO','ND','NE','OH','SD','WI') THEN 'Midwest'
        WHEN property_state IN ('AZ','NM','OK','TX') THEN 'Southwest'
        WHEN property_state IN ('AK','CA','CO','HI','ID','MT','NV','OR','UT','WA','WY') THEN 'West'
        WHEN property_state IN ('DC','MD') THEN 'DC/MD'
        ELSE 'Other'
    END                                                         AS state_group,

    -- ── Aggregated Metrics ─────────────────────────────────────────────
    COUNT(DISTINCT loan_id)                                     AS loan_count,
    SUM(current_upb)                                            AS upb,

    -- WA metrics
    SUM(current_interest_rate * current_upb)
        / NULLIF(SUM(current_upb), 0)                           AS wac,
    SUM(fico * current_upb)
        / NULLIF(SUM(CASE WHEN fico IS NOT NULL THEN current_upb ELSE 0 END), 0)
                                                                AS wa_fico,
    SUM(original_ltv * current_upb)
        / NULLIF(SUM(CASE WHEN original_ltv IS NOT NULL THEN current_upb ELSE 0 END), 0)
                                                                AS wa_ltv,
    SUM(dti * current_upb)
        / NULLIF(SUM(CASE WHEN dti IS NOT NULL THEN current_upb ELSE 0 END), 0)
                                                                AS wa_dti,
    SUM(loan_age * current_upb)
        / NULLIF(SUM(current_upb), 0)                           AS wala,

    -- Pool % (will be normalised post-aggregation)
    SUM(current_upb)                                            AS pool_upb_raw,

    -- Delinquency rate
    100.0 * SUM(CASE WHEN delinquency_months > 0 THEN current_upb ELSE 0 END)
        / NULLIF(SUM(current_upb), 0)                           AS delinquency_rate_pct

FROM silver.loan_month_panel
WHERE is_terminal = FALSE
  AND reporting_period = (
      SELECT MAX(reporting_period)
      FROM silver.loan_month_panel
      WHERE is_terminal = FALSE
  )
GROUP BY 1, 2, 3, 4, 5
HAVING COUNT(DISTINCT loan_id) >= 5
ORDER BY vintage, fico_band, ltv_band, coupon_band, state_group
;
