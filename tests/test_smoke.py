import pandas as pd
from medq import Columns, Config, DataLoader, TextCleaner, QualityFixer


def _make_files(folder):
    pd.DataFrame({"id": [1], "question": ["سؤال"], "category": ["أ"]}).to_csv(folder / "train.csv", index=False)
    pd.DataFrame({"id": [2], "question": ["سؤال"], "category": ["أ"]}).to_csv(folder / "validation.csv", index=False)
    pd.DataFrame({"id": [3], "question": [None]}).to_csv(folder / "test 1.csv", index=False)
    pd.DataFrame({"Category": ["أ"], "Label ID": [0]}).to_csv(folder / "labels_mapping.csv", index=False)


def test_loader_finds_test_with_space(tmp_path):
    _make_files(tmp_path)
    ds = DataLoader(Config(data_dir=tmp_path)).load()
    assert len(ds.train) == 1 and len(ds.test) == 1


def test_cleaner_survives_nan_and_adds_column(tmp_path):
    _make_files(tmp_path)
    ds = DataLoader(Config(data_dir=tmp_path)).load()
    out = TextCleaner("light").transform(ds.test)
    assert Columns.CLEAN in out.columns and out[Columns.CLEAN].iloc[0] == ""


def test_quality_never_drops_test_rows(tmp_path):
    _make_files(tmp_path)
    ds = DataLoader(Config(data_dir=tmp_path)).load()
    assert len(QualityFixer().apply(ds.test, is_train=False)) == len(ds.test)
