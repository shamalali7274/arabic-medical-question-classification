"""تحميل البيانات: كلاس DataLoader يرجع Dataset جاهز."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config


@dataclass
class Dataset:
    """حاوية للأجزاء الأربعة، بدل ما نرجع tuple ونتلخبط بالترتيب."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_map: pd.DataFrame


class DataLoader:
    def __init__(self, config: Config = None):
        self.config = config or Config()

    def load(self) -> Dataset:
        return Dataset(
            train=pd.read_csv(self._find("train.csv")),
            val=pd.read_csv(self._find("validation.csv")),
            test=pd.read_csv(self._find("test.csv", "test 1.csv")),
            label_map=self._load_label_map(),
        )

    def load_sample_submission(self) -> pd.DataFrame:
        return pd.read_csv(self._find("sample_submission.csv"))

    def _load_label_map(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self._find("label_mapping.csv", "labels_mapping.csv"))
        except FileNotFoundError:
            return pd.read_excel(self._find("labels_mapping.xlsx"))

    def _find(self, *candidates: str) -> Path:
        """أول اسم موجود من القائمة (لأن test 1.csv مثلاً اسمه يختلف عن الوظيفة)."""
        for name in candidates:
            path = self.config.data_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"ما لقيت أياً من {candidates} داخل {self.config.data_dir}")
