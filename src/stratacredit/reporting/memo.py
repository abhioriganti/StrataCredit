"""HTML deal analytics memo generator.

Generates a self-contained HTML analytical memo from deterministic
calculations.  No LLM required — all observations are computed from data.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from jinja2 import Template

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>StrataCredit Deal Analytics Memo</title>
<style>
  body{font-family:'Segoe UI',Arial,sans-serif;max-width:960px;margin:40px auto;
       padding:0 20px;color:#1a1a2e;background:#f8f9fa}
  h1{color:#0d47a1;border-bottom:3px solid #0d47a1;padding-bottom:8px}
  h2{color:#1565c0;margin-top:2em;border-bottom:1px solid #90caf9}
  .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
  .card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
  .lbl{font-size:.75rem;color:#666;text-transform:uppercase;letter-spacing:.05em}
  .val{font-size:1.4rem;font-weight:700;color:#0d47a1;margin-top:4px}
  table{width:100%;border-collapse:collapse;margin:16px 0;font-size:.9rem}
  thead{background:#1565c0;color:#fff}
  th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #e0e0e0}
  tr:nth-child(even){background:#f0f4ff}
  .pass{color:#2e7d32;font-weight:bold}
  .fail{color:#c62828;font-weight:bold}
  .obs{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;
       margin:12px 0;border-radius:0 6px 6px 0;font-style:italic}
  .warn{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px;margin:10px 0}
  .disc{background:#fbe9e7;border:1px solid #ef9a9a;padding:14px;
        border-radius:6px;font-size:.82rem;margin-top:2em}
  footer{color:#999;font-size:.8rem;margin-top:2em;padding-top:1em;
         border-top:1px solid #ddd}
</style>
</head>
<body>
<h1>StrataCredit Deal Analytics Memo</h1>
<p><strong>As-of Date:</strong> {{as_of}} &nbsp;|&nbsp;
   <strong>Generated:</strong> {{generated}}</p>

<h2>1. Collateral Overview</h2>
<div class="grid">
  <div class="card"><div class="lbl">Pool UPB</div>
    <div class="val">${{"%.1f"|format(m.total_current_upb/1e6)}}M</div></div>
  <div class="card"><div class="lbl">Loan Count</div>
    <div class="val">{{"{:,}".format(m.loan_count)}}</div></div>
  <div class="card"><div class="lbl">WA Coupon</div>
    <div class="val">{{"%.2f"|format(m.wa_coupon or 0)}}%</div></div>
  <div class="card"><div class="lbl">WA FICO</div>
    <div class="val">{{"%.0f"|format(m.wa_fico or 0)}}</div></div>
  <div class="card"><div class="lbl">WA LTV</div>
    <div class="val">{{"%.1f"|format(m.wa_ltv or 0)}}%</div></div>
  <div class="card"><div class="lbl">WA DTI</div>
    <div class="val">{{"%.1f"|format(m.wa_dti or 0)}}%</div></div>
  <div class="card"><div class="lbl">WALA</div>
    <div class="val">{{"%.0f"|format(m.wala or 0)}} mo</div></div>
  <div class="card"><div class="lbl">Purchase %</div>
    <div class="val">{{"%.1f"|format(m.purchase_pct)}}%</div></div>
  <div class="card"><div class="lbl">Investor %</div>
    <div class="val">{{"%.1f"|format(m.investor_pct)}}%</div></div>
</div>
<div class="obs">{{observation}}</div>

<h2>2. Key Concentrations</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
  <tr><td>Low-FICO (&lt;660)</td><td>{{"%.1f"|format(m.low_fico_pct)}}%</td></tr>
  <tr><td>High-LTV (&gt;80%)</td><td>{{"%.1f"|format(m.high_ltv_pct)}}%</td></tr>
  <tr><td>Modified</td><td>{{"%.1f"|format(m.modified_pct)}}%</td></tr>
  <tr><td>Largest State ({{m.largest_state}})</td>
      <td>{{"%.1f"|format(m.largest_state_pct)}}%</td></tr>
  <tr><td>DQ30+</td>
      <td>{{"%.2f"|format(m.dq30_pct+m.dq60_pct+m.dq90plus_pct)}}%</td></tr>
</tbody></table>

<h2>3. Pool Constraint Reconciliation</h2>
<table>
<thead><tr><th>Constraint</th><th>Target</th><th>Actual</th><th>Status</th></tr></thead>
<tbody>
{% for r in constraints %}
<tr><td>{{r.name}}</td>
    <td>{{r.unit}}{{"%.1f"|format(r.target) if r.target is not none else "—"}}</td>
    <td>{{r.unit}}{{"%.2f"|format(r.actual)}}</td>
    <td><span class="{{'pass' if r.status=='PASS' else 'fail'}}">{{r.status}}</span></td>
</tr>
{% endfor %}
</tbody></table>

<h2>4. Stress Scenarios</h2>
<div class="warn">Deterministic sensitivity analysis only. Not a rating-agency model.</div>
<table>
<thead><tr><th>Scenario</th><th>CE Mult</th><th>Sev Mult</th>
  <th>Annual Gross Loss %</th><th>Cum 3yr %</th><th>Cum 5yr %</th></tr></thead>
<tbody>
{% for s in scenarios %}
<tr><td><strong>{{s.scenario_label}}</strong></td>
    <td>{{s.default_multiplier}}×</td><td>{{s.severity_multiplier}}×</td>
    <td>{{"%.3f"|format(s.projected_gross_loss_pct)}}%</td>
    <td>{{"%.3f"|format(s.cumulative_loss_pct_3yr)}}%</td>
    <td>{{"%.3f"|format(s.cumulative_loss_pct_5yr)}}%</td>
</tr>
{% endfor %}
</tbody></table>

<h2>5. Methodology Note</h2>
<p>CPR uses standard SMM methodology with scheduled principal adjustment.
CDR is StrataCredit's internal definition and does not replicate any
rating-agency methodology. Loss severity = actual_loss / zero_balance_removal_upb
from Freddie Mac servicer data.</p>

<div class="disc"><strong>Limitations:</strong> Analytics based on Freddie Mac
Single-Family Loan-Level Dataset (research license). For informational purposes
only. Not investment advice. Not a credit rating. Past performance does not
predict future results. Raw data may not be redistributed.</div>
<footer>StrataCredit v0.1.0 | {{generated}}</footer>
</body></html>
"""


def _make_observation(m, cand_m=None) -> str:
    parts = []
    if cand_m and m.wa_fico and cand_m.wa_fico:
        d = m.wa_fico - cand_m.wa_fico
        if abs(d) > 1:
            verb = "increased" if d > 0 else "decreased"
            parts.append(
                f"The selected pool {verb} WA FICO from {cand_m.wa_fico:.0f} to {m.wa_fico:.0f}"
            )
    if cand_m and m.high_ltv_pct is not None and cand_m.high_ltv_pct is not None:
        d = m.high_ltv_pct - cand_m.high_ltv_pct
        if abs(d) > 0.5:
            verb = "reducing" if d < 0 else "increasing"
            parts.append(
                f"{verb} high-LTV from {cand_m.high_ltv_pct:.1f}% to {m.high_ltv_pct:.1f}%"
            )
    if not parts:
        parts.append(
            f"Selected pool: {m.loan_count:,} loans, ${m.total_current_upb / 1e6:.1f}M UPB"
        )
    return ". ".join(parts) + "."


def generate_memo(
    metrics,
    constraint_results: list,
    scenario_results: list,
    as_of_date=None,
    candidate_metrics=None,
    output_path: Path | None = None,
) -> Path:
    """Generate HTML deal analytics memo.

    Returns path to the generated HTML file.
    """
    out = output_path or Path("outputs") / "deal_analytics_memo.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    html = Template(_TEMPLATE).render(
        m=metrics,
        constraints=constraint_results,
        scenarios=scenario_results,
        observation=_make_observation(metrics, candidate_metrics),
        as_of=str(as_of_date or date.today()),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    out.write_text(html, encoding="utf-8")
    return out
