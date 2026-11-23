"""إعدادات المشروع المشتركة: كل الثوابت (أسماء الأعمدة، المسارات، الرقم العشوائي) بمكان واحد."""
from dataclasses import dataclass
from pathlib import Path

KAGGLE_ROOT = Path("/kaggle/input")
LOCAL_DATA_DIR = Path("data")


class Columns:
    """أسماء الأعمدة. ممنوع كتابة 'question' أو 'category' كنص مباشر بباقي الكود."""
    ID = "id"
    TEXT = "question"
    LABEL = "category"
    CLEAN = "clean_text"


def find_data_dir() -> Path:
    """على Kaggle: يدوّر على المجلد اللي فيه train.csv. محلياً: يرجع data/."""
    if KAGGLE_ROOT.exists():
        for train_file in KAGGLE_ROOT.rglob("train.csv"):
            return train_file.parent
    return LOCAL_DATA_DIR


@dataclass(frozen=True)
class Config:
    """frozen=True: ما بنقدر نغيّر القيم بعد الإنشاء، فما حدا يخرّب الإعدادات بالغلط."""
    data_dir: Path = None
    random_state: int = 42

    def __post_init__(self):
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", find_data_dir())
