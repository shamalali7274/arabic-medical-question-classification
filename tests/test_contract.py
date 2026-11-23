import numpy as np
import pandas as pd
import pytest

from common import Columns
from preprocess import apply_clean, clean_text
from quality import apply_quality_fixes


def _df():
    return pd.DataFrame({"id": [1, 2], "question": ["سؤال", None], "category": ["أ", "ب"]})


def test_quality_placeholder_never_drops_test_rows():
    df = _df().drop(columns="category")
    assert len(apply_quality_fixes(df, is_train=False)) == len(df)


@pytest.mark.parametrize("mode", ["none", "light", "heavy"])
def test_clean_text_accepts_all_modes_and_nan(mode):
    assert clean_text(np.nan, mode) == ""
    assert isinstance(clean_text("سؤال", mode), str)


def test_clean_text_rejects_unknown_mode():
    with pytest.raises(ValueError):
        clean_text("سؤال", "extreme")


def test_apply_clean_adds_clean_text_column_without_touching_input():
    df = _df()
    out = apply_clean(df, "light")
    assert Columns.CLEAN in out.columns and Columns.CLEAN not in df.columns
    assert len(out) == len(df)
