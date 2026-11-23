import pandas as pd
import pytest

from common import Columns, DataLoader, load_data


def _write(folder, test_name="test.csv", label_name="label_mapping.csv", label_cols=("category", "label_id")):
    pd.DataFrame({"id": [1, 2], "question": ["س1", "س2"], "category": ["أ", "ب"]}).to_csv(folder / "train.csv", index=False)
    pd.DataFrame({"id": [3], "question": ["س3"], "category": ["أ"]}).to_csv(folder / "validation.csv", index=False)
    pd.DataFrame({"id": [4], "question": [None]}).to_csv(folder / test_name, index=False)
    labels = pd.DataFrame({label_cols[0]: ["أ", "ب"], label_cols[1]: [0, 1]})
    if label_name.endswith(".xlsx"):
        labels.to_excel(folder / label_name, index=False)
    else:
        labels.to_csv(folder / label_name, index=False)


def test_load_standard_names(tmp_path):
    _write(tmp_path)
    train, val, test, label_map = load_data(tmp_path)
    assert (len(train), len(val), len(test)) == (2, 1, 1)
    assert list(label_map.columns) == [Columns.LABEL, Columns.LABEL_ID]


def test_test_file_with_space_and_xlsx_label_map(tmp_path):
    _write(tmp_path, test_name="test 1.csv", label_name="labels_mapping.xlsx", label_cols=("Category", "Label ID"))
    _, _, test, label_map = load_data(tmp_path)
    assert len(test) == 1
    assert list(label_map.columns) == [Columns.LABEL, Columns.LABEL_ID]


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataLoader(tmp_path).load()


def test_missing_column_is_reported(tmp_path):
    _write(tmp_path)
    pd.DataFrame({"id": [1], "text": ["x"], "category": ["أ"]}).to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError, match="question"):
        DataLoader(tmp_path).load()
