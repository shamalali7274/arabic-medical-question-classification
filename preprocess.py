"""Text cleaning. PLACEHOLDER written by Person 1; Person 2 replaces the bodies.

Contract:
    clean_text(text, mode)  mode in {"none", "light", "heavy"}; must accept NaN
    apply_clean(df, mode)   returns a copy of df with a ``clean_text`` column
"""
from __future__ import annotations

from enum import Enum

import pandas as pd

from common import Columns


class CleanMode(str, Enum):
    NONE = "none"
    LIGHT = "light"
    HEAVY = "heavy"


class TextCleaner:
    """Placeholder: returns the text unchanged (and '' for NaN)."""

    def __init__(self, mode: CleanMode | str = CleanMode.LIGHT) -> None:
        self.mode = CleanMode(mode)  # raises ValueError for unknown modes

    def clean(self, text) -> str:
        if pd.isna(text):
            return ""
        return str(text)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[Columns.CLEAN] = out[Columns.TEXT].map(self.clean)
        return out


def clean_text(text, mode: CleanMode | str = "light") -> str:
    return TextCleaner(mode).clean(text)


def apply_clean(df: pd.DataFrame, mode: CleanMode | str = "light") -> pd.DataFrame:
    return TextCleaner(mode).transform(df)
