"""Data-quality fixes. PLACEHOLDER written by Person 1; Person 2 replaces the body.

Contract: ``apply_quality_fixes(df, is_train)`` returns a DataFrame.
Rule: rows may be dropped or labels unified only when ``is_train`` is True
(train and validation). Test rows are NEVER dropped.
"""
from __future__ import annotations

import pandas as pd


class QualityFixer:
    """Placeholder: returns the data unchanged."""

    def apply(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        return df.copy()


def apply_quality_fixes(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    return QualityFixer().apply(df, is_train)
