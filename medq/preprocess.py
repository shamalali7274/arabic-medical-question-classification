"""تنظيف النص (الشخص 2). الآن نسخة وهمية ترجع النص كما هو."""
from enum import Enum

import pandas as pd

from .config import Columns


class CleanMode(str, Enum):
    NONE = "none"
    LIGHT = "light"
    HEAVY = "heavy"


class TextCleaner:
    def __init__(self, mode: CleanMode = CleanMode.LIGHT):
        self.mode = CleanMode(mode)  # يقبل "light" أو CleanMode.LIGHT، ويرفض أي شي ثاني

    def clean(self, text) -> str:
        """تتحمل NaN بدون ما تنهار."""
        if pd.isna(text):
            return ""
        return str(text)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """ترجع نسخة من الجدول مع عمود clean_text."""
        out = df.copy()
        out[Columns.CLEAN] = out[Columns.TEXT].map(self.clean)
        return out
