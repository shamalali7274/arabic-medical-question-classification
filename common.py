"""Shared configuration and data loading: the team contract (Person 1 owns this file).

Public API (do not rename without telling the team):
    DATA_DIR        folder that holds the CSV files (Kaggle, env var, or ./data)
    RANDOM_STATE    shared random seed (42)
    Columns         canonical column names
    load_data()     -> (train, val, test, label_map)

Everything else in this module is the OOP machinery behind that API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

RANDOM_STATE = 42


class Columns:
    """Canonical column names. Never hard-code 'question' or 'category' elsewhere."""

    ID = "id"
    TEXT = "question"
    LABEL = "category"
    LABEL_ID = "label_id"
    CLEAN = "clean_text"


class DataDirectoryFinder:
    """Locate the data folder: env var > Kaggle input > local ``data/``."""

    ENV_VAR = "MEDQ_DATA_DIR"
    KAGGLE_ROOT = Path("/kaggle/input")
    LOCAL_DIR = Path(__file__).resolve().parent / "data"  # <repo>/data, whatever the working directory
    MARKER_FILE = "train.csv"

    @classmethod
    def find(cls) -> Path:
        from_env = os.environ.get(cls.ENV_VAR)
        if from_env:
            return Path(from_env)
        if cls.KAGGLE_ROOT.exists():
            for marker in cls.KAGGLE_ROOT.rglob(cls.MARKER_FILE):
                return marker.parent
        return cls.LOCAL_DIR


DATA_DIR = DataDirectoryFinder.find()


@dataclass
class Dataset:
    """The four tables of the project, with names instead of tuple positions."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_map: pd.DataFrame

    @property
    def splits(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def as_tuple(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.train, self.val, self.test, self.label_map


class DataLoader:
    """Read the project files, tolerating the naming differences we know about.

    * ``test.csv`` may be called ``test 1.csv``.
    * the label map may be ``label_mapping.csv``, ``labels_mapping.csv`` or ``labels_mapping.xlsx``.
    * the label map's columns may be ``category/label_id`` or ``Category/Label ID``;
      they are always returned as ``category`` and ``label_id``.
    """

    TRAIN_NAMES = ("train.csv",)
    VAL_NAMES = ("validation.csv", "val.csv")
    TEST_NAMES = ("test.csv", "test 1.csv")
    SAMPLE_NAMES = ("sample_submission.csv",)
    LABEL_MAP_NAMES = ("label_mapping.csv", "labels_mapping.csv", "labels_mapping.xlsx", "label_mapping.xlsx")

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    def load(self) -> Dataset:
        train = self._read_labelled(self.TRAIN_NAMES, labelled=True)
        val = self._read_labelled(self.VAL_NAMES, labelled=True)
        test = self._read_labelled(self.TEST_NAMES, labelled=False)
        return Dataset(train=train, val=val, test=test, label_map=self._read_label_map())

    def load_sample_submission(self) -> pd.DataFrame:
        return self._read_table(self._find(self.SAMPLE_NAMES))

    # ------------------------------------------------------------------ helpers
    def _read_labelled(self, names: Iterable[str], labelled: bool) -> pd.DataFrame:
        df = self._read_table(self._find(names))
        required = [Columns.ID, Columns.TEXT] + ([Columns.LABEL] if labelled else [])
        self._require_columns(df, required, names)
        return df

    def _read_label_map(self) -> pd.DataFrame:
        df = self._read_table(self._find(self.LABEL_MAP_NAMES))
        df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
        self._require_columns(df, [Columns.LABEL, Columns.LABEL_ID], self.LABEL_MAP_NAMES)
        return df[[Columns.LABEL, Columns.LABEL_ID]]

    def _find(self, candidates: Iterable[str]) -> Path:
        candidates = tuple(candidates)
        for name in candidates:
            path = self.data_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of {candidates} found in {self.data_dir.resolve()}")

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path, encoding="utf-8-sig")

    @staticmethod
    def _require_columns(df: pd.DataFrame, required: list[str], source) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {source}; found {list(df.columns)}")


def load_data(data_dir: Path | str | None = None):
    """Contract function: returns ``train, val, test, label_map``."""
    return DataLoader(data_dir).load().as_tuple()
