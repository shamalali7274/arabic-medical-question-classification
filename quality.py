"""إصلاحات جودة البيانات (الشخص 2). كل مشكلة لها دالة find_* تعرض المشكلة، و apply تطبّق القرارات.

دالة العقد: apply_quality_fixes(df, is_train, reference=None) (بآخر الملف).
القرارات النهائية مكتوبة بمكان واحد: DECISIONS.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from common import Columns, load_data
from preprocess import CleanOptions, TextCleaner

# قراراتي بعد قراءة الأرقام (الدفتر notebooks/02_quality.ipynb بيستخدم نفس القاموس)
DECISIONS = dict(
    unify_labels=True,
    min_words=1,
    min_arabic_ratio=0.0,
    drop_conflicts=True,
    drop_duplicates=True,
    drop_leaked=True,
)

_LETTER = re.compile(r"[^\W\d_]")
_ARABIC_LETTER = re.compile(r"[\u0621-\u064a]")
_KEY = "_key"


def arabic_ratio(text) -> float:
    """نسبة الحروف العربية من كل الحروف بالنص (0 إذا ما في حروف)."""
    letters = _LETTER.findall("" if pd.isna(text) else str(text))
    if not letters:
        return 0.0
    return len(_ARABIC_LETTER.findall("".join(letters))) / len(letters)


class QualityFixer:
    """
    label_names: أسماء الفئات الرسمية من ملف الترميز (اختياري). إذا انعطت، توحيد الفئات بيرجع لأسمائها.
                 إذا ما انعطت، بتتعلم الأسماء من أول جدول بينعطى لـ apply (نادي apply على train أولاً).
    min_words: أي سؤال كلماته أقل من هالرقم بينحذف (1 = حذف الفاضي فقط).
    min_arabic_ratio: أي سؤال نسبة حروفه العربية أقل منها بينحذف (0 = معطّل).
    drop_conflicts: حذف كل الأسئلة اللي نفس نصها (بعد التطبيع) إلها فئتين مختلفتين.
    drop_duplicates: حذف التكرار (نفس النص بعد التطبيع ونفس الفئة) مع الإبقاء على أول نسخة.
    drop_leaked: حذف الأسئلة الموجودة بجدول reference (مثلاً حذف أسئلة validation الموجودة بـ train).
    """

    def __init__(self, label_names=None, unify_labels=True, min_words=1, min_arabic_ratio=0.0,
                 drop_conflicts=True, drop_duplicates=True, drop_leaked=False):
        self.label_names = list(label_names) if label_names is not None else None
        self.unify_labels = unify_labels
        self.min_words = min_words
        self.min_arabic_ratio = min_arabic_ratio
        self.drop_conflicts = drop_conflicts
        self.drop_duplicates = drop_duplicates
        self.drop_leaked = drop_leaked
        self._key_cleaner = TextCleaner(options=CleanOptions.for_key())
        self._canonical = None
        self.log_ = []

    # ---------- مفاتيح المقارنة ----------
    def text_key(self, texts: pd.Series) -> pd.Series:
        return texts.map(self._key_cleaner.clean)

    def label_key(self, label) -> str:
        return self._key_cleaner.clean(label)

    def _with_key(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{_KEY: self.text_key(df[Columns.TEXT])})

    # ---------- كشف المشاكل (للدفتر) ----------
    def find_short(self, df: pd.DataFrame, min_words: int = None) -> pd.Series:
        """True للسؤال الفاضي (NaN أو مسافات فقط) أو اللي كلماته أقل من min_words."""
        threshold = self.min_words if min_words is None else min_words
        words = df[Columns.TEXT].fillna("").astype(str).str.split().str.len()
        return words < threshold

    def find_non_arabic(self, df: pd.DataFrame, min_ratio: float = None) -> pd.Series:
        threshold = self.min_arabic_ratio if min_ratio is None else min_ratio
        return df[Columns.TEXT].map(arabic_ratio) < threshold

    def label_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """الفئات اللي إلها أكثر من شكل كتابة: لكل مفتاح موحّد، الأشكال وعدد أسئلة كل شكل."""
        counts = df[Columns.LABEL].value_counts().rename("count").rename_axis("label").reset_index()
        counts["key"] = counts["label"].map(self.label_key)
        multi = counts.groupby("key")["label"].transform("nunique") > 1
        return counts[multi].sort_values(["key", "count"], ascending=[True, False]).reset_index(drop=True)

    def find_duplicates(self, df: pd.DataFrame, same_label: bool = True) -> pd.DataFrame:
        """كل الصفوف المكررة (بما فيها النسخة الأولى) مرتبة حسب النص."""
        keyed = self._with_key(df)
        subset = [_KEY, Columns.LABEL] if same_label else [_KEY]
        return keyed[keyed.duplicated(subset, keep=False)].sort_values(_KEY)

    def find_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """الأسئلة اللي نفس نصها (بعد التطبيع) إلها أكثر من فئة، مرتبة حسب النص."""
        keyed = self._with_key(df)
        return keyed[self._conflict_mask(keyed)].sort_values(_KEY)

    def find_leaked(self, df: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
        """True للسؤال اللي نصه (بعد التطبيع) موجود بـ reference."""
        reference_keys = set(self.text_key(reference[Columns.TEXT]))
        return self.text_key(df[Columns.TEXT]).isin(reference_keys)

    # ---------- توحيد الفئات ----------
    def fit_labels(self, df: pd.DataFrame) -> "QualityFixer":
        """بتحدد الاسم الرسمي لكل فئة: من label_names إذا موجودة، وإلا أكثر شكل تكراراً بالجدول."""
        if self.label_names is not None:
            self._canonical = {self.label_key(name): name for name in self.label_names}
        else:
            counts = df[Columns.LABEL].dropna().value_counts()
            self._canonical = {}
            for label in counts.index:
                self._canonical.setdefault(self.label_key(label), label)
        return self

    def unify_label_names(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._canonical is None:
            self.fit_labels(df)
        out = df.copy()
        keys = out[Columns.LABEL].map(lambda label: label if pd.isna(label) else self.label_key(label))
        unified = keys.map(self._canonical)
        out[Columns.LABEL] = unified.fillna(out[Columns.LABEL])
        return out

    # ---------- التطبيق ----------
    def apply(self, df: pd.DataFrame, is_train: bool, reference: pd.DataFrame = None) -> pd.DataFrame:
        """is_train=True: مسموح الحذف والتوحيد. is_train=False (test): ممنوع حذف أي صف.

        توحيد أسماء الفئات ما فيه حذف، فبيتطبّق دائماً إذا عمود الفئة موجود (حتى مع is_train=False)،
        وبهيك validation ما بيضل بـ 14 اسم حتى لو انستدعى بالغلط بـ is_train=False.
        """
        out = df.copy()
        out[Columns.TEXT] = out[Columns.TEXT].fillna("")
        if not is_train:
            if self.unify_labels and Columns.LABEL in out.columns:
                out = self.unify_label_names(out)
            return out

        self.log_ = [("البداية", len(out))]

        removed = self.find_short(out) | out[Columns.LABEL].isna()
        out = self._drop(out, removed, "حذف الفاضي والقصير")

        if self.min_arabic_ratio > 0:
            out = self._drop(out, self.find_non_arabic(out), "حذف النص غير العربي")

        if self.unify_labels:
            out = self.unify_label_names(out)
            self._log("توحيد أسماء الفئات", out)

        keyed = self._with_key(out)
        if self.drop_conflicts:
            keyed = self._drop(keyed, self._conflict_mask(keyed), "حذف تضارب الملصقات")
        if self.drop_duplicates:
            keyed = self._drop(keyed, keyed.duplicated([_KEY, Columns.LABEL]), "حذف التكرار")
        if self.drop_leaked and reference is not None:
            leaked = keyed[_KEY].isin(set(self.text_key(reference[Columns.TEXT])))
            keyed = self._drop(keyed, leaked, "حذف التسريب من reference")

        return keyed.drop(columns=_KEY).reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """جدول: كل خطوة وعدد الصفوف بعدها وكم صف حذفت."""
        table = pd.DataFrame(self.log_, columns=["step", "rows"])
        table["removed"] = (table["rows"].shift(1) - table["rows"]).fillna(0).astype(int)
        return table

    # ---------- مساعدات داخلية ----------
    @staticmethod
    def _conflict_mask(keyed: pd.DataFrame) -> pd.Series:
        return keyed.groupby(_KEY)[Columns.LABEL].transform("nunique") > 1

    def _drop(self, df: pd.DataFrame, mask: pd.Series, step: str) -> pd.DataFrame:
        out = df[~mask]
        self._log(step, out)
        return out

    def _log(self, step: str, df: pd.DataFrame) -> None:
        self.log_.append((step, len(df)))


@lru_cache(maxsize=1)
def _official_label_names() -> tuple:
    """أسماء الفئات الرسمية من ملف الترميز (بتنقرأ مرة وحدة)."""
    return tuple(load_data()[3][Columns.LABEL])


def apply_quality_fixes(df: pd.DataFrame, is_train: bool, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    
    if not is_train:
        names = _official_label_names() if Columns.LABEL in df.columns else None
        return QualityFixer(label_names=names).apply(df, is_train=False)
    fixer = QualityFixer(label_names=_official_label_names(), **DECISIONS)
    return fixer.apply(df, is_train=True, reference=reference)