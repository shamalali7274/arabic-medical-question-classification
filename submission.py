"""Build and validate ``submission.csv`` (Person 1, Wednesday task).

Usage (from the final notebook):
    builder = SubmissionBuilder(sample_submission, label_map)
    submission = builder.build(test, predictions)      # names or label ids
    builder.save(submission, "submission.csv")

The result has exactly the rows, ids and id order of ``sample_submission.csv`` and
category *names* taken from ``label_mapping`` (never raw spelling variants).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from common import Columns


class SubmissionError(ValueError):
    """Raised when a submission would not match the required format."""


class SubmissionBuilder:
    def __init__(self, sample_submission: pd.DataFrame, label_map: pd.DataFrame) -> None:
        self.sample = sample_submission
        self.label_map = label_map
        self._id_to_name = dict(zip(label_map[Columns.LABEL_ID], label_map[Columns.LABEL]))
        self._names = set(label_map[Columns.LABEL])

    def build(self, test: pd.DataFrame, predictions: Sequence) -> pd.DataFrame:
        """Align predictions (names or label ids, in ``test`` row order) to the sample's id order."""
        if len(predictions) != len(test):
            raise SubmissionError(f"{len(predictions)} predictions for {len(test)} test rows")
        labels = pd.Series(list(predictions), index=test[Columns.ID].to_numpy())
        labels = labels.map(lambda p: self._id_to_name.get(p, p))
        submission = pd.DataFrame({Columns.ID: self.sample[Columns.ID].to_numpy()})
        submission[Columns.LABEL] = labels.reindex(submission[Columns.ID]).to_numpy()
        self.validate(submission)
        return submission

    def validate(self, submission: pd.DataFrame) -> None:
        """Raise ``SubmissionError`` unless the file matches the required format."""
        if list(submission.columns) != [Columns.ID, Columns.LABEL]:
            raise SubmissionError(f"columns must be [id, category], got {list(submission.columns)}")
        if len(submission) != len(self.sample):
            raise SubmissionError(f"{len(submission)} rows, expected {len(self.sample)}")
        if not submission[Columns.ID].equals(self.sample[Columns.ID]):
            raise SubmissionError("ids or their order differ from sample_submission")
        if submission[Columns.LABEL].isna().any():
            raise SubmissionError(f"{int(submission[Columns.LABEL].isna().sum())} rows have no category")
        unknown = set(submission[Columns.LABEL]) - self._names
        if unknown:
            raise SubmissionError(f"categories not in label_mapping: {sorted(unknown)}")

    def save(self, submission: pd.DataFrame, path: str | Path = "submission.csv") -> Path:
        self.validate(submission)
        path = Path(path)
        submission.to_csv(path, index=False, encoding="utf-8")
        return path
