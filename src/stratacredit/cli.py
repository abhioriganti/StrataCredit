"""Small command-line interface for environments without Make."""

from __future__ import annotations

import click

from stratacredit.analytics.runner import run as _analytics
from stratacredit.ingest.runner import run as _ingest
from stratacredit.models.full_trainer import run as _train
from stratacredit.models.leakage_audit import run as _leakage_audit
from stratacredit.models.mlflow_tracking import register_runs as _mlflow
from stratacredit.models.scoring import persist_scores as _score
from stratacredit.models.xgb_trainer import run as _xgb
from stratacredit.optimization.runner import run as _pool
from stratacredit.quality.runner import run as _validate
from stratacredit.reporting.runner import run as _report
from stratacredit.scenarios.runner import run as _scenarios
from stratacredit.transform.runner import run as _transform


@click.group()
def main() -> None:
    """StrataCredit analytical workflow."""


def _command(name: str, func) -> None:
    main.command(name)(click.pass_context(lambda ctx: func()))


for _name, _function in {
    "ingest": _ingest,
    "validate": _validate,
    "transform": _transform,
    "analytics": _analytics,
    "train": _train,
    "xgb": _xgb,
    "score": _score,
    "mlflow": _mlflow,
    "leakage-audit": _leakage_audit,
    "pool": _pool,
    "scenarios": _scenarios,
    "report": _report,
}.items():
    _command(_name, _function)
