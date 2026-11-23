import pandas as pd
import pytest

from submission import SubmissionBuilder, SubmissionError


def _fixtures():
    label_map = pd.DataFrame({"category": ["أ", "ب"], "label_id": [0, 1]})
    sample = pd.DataFrame({"id": [3, 1, 2], "category": [None, None, None]})
    test = pd.DataFrame({"id": [1, 2, 3], "question": ["a", "b", "c"]})
    return SubmissionBuilder(sample, label_map), test


def test_build_reorders_to_sample_ids_and_accepts_label_ids():
    builder, test = _fixtures()
    sub = builder.build(test, [0, 1, 1])          # ids 1,2,3 -> أ,ب,ب
    assert sub["id"].tolist() == [3, 1, 2]
    assert sub["category"].tolist() == ["ب", "أ", "ب"]


def test_build_accepts_names():
    builder, test = _fixtures()
    assert builder.build(test, ["أ", "أ", "ب"])["category"].tolist() == ["ب", "أ", "أ"]


def test_wrong_length_and_unknown_category_are_rejected():
    builder, test = _fixtures()
    with pytest.raises(SubmissionError):
        builder.build(test, [0, 1])
    with pytest.raises(SubmissionError):
        builder.build(test, ["أ", "أ", "غير موجودة"])


def test_save_writes_csv(tmp_path):
    builder, test = _fixtures()
    path = builder.save(builder.build(test, [0, 1, 0]), tmp_path / "s.csv")
    assert list(pd.read_csv(path).columns) == ["id", "category"]
