"""تنظيف النص (الشخص 2): كل خطوة تنظيف خيار مستقل، و light/heavy مجرد إعدادات جاهزة منها.

دوال العقد: clean_text(text, mode) و apply_clean(df, mode) (بآخر الملف).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum

import pandas as pd

from common import Columns

_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL = "\u0640"
_ALEF = str.maketrans("أإآٱ", "اااا")
_YEH = str.maketrans("ى", "ي")
_TEH_MARBUTA = str.maketrans("ة", "ه")
_DIGITS = str.maketrans(
    {chr(base + i): str(i) for base in (0x0660, 0x06F0) for i in range(10)}
)
_NUMBER = re.compile(r"(?<![A-Za-z_\d])\d+(?:[.,٫]\d+)?(?![A-Za-z_\d]|[.,٫]\d)")
_REPEATED_LETTER = re.compile(r"([^\W\d_])\1{2,}")
_PUNCTUATION = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")

NUMBER_TOKEN = "NUM"

# كلمات موجودة بقائمة NLTK لكنها تحمل معلومة طبية، فلا نحذفها (المراجعة اليدوية بقسم 11 بالدفتر):
# نفي، زمن، كلمات هي نفسها عضو أو مرض، وكلمات ارتبطت بفئة معينة بالبيانات
# (أسماء الأشهر: تواريخ الدورة والحمل، والأرقام: عمر الطفل بالأشهر وشهر الحمل، ود/دال: فيتامين د، وكلمات النظر).
NEGATION_WORDS = frozenset({
    "لا", "لم", "لن", "ما", "ليس", "ليست", "ليسا", "ليسوا", "لست", "غير", "دون", "عدم", "بدون", "بلا",
    "ولا", "وما", "فلا",
})
TIME_WORDS = frozenset({"قبل", "بعد", "منذ", "مذ", "أمس", "غدا", "الآن", "صباح", "مساء", "مرة"})
BODY_WORDS = frozenset({"عين", "نفس", "ذات", "حار", "خال"})
MONTH_WORDS = frozenset({
    "يناير", "فبراير", "مارس", "ابريل", "أبريل", "مايو", "يونيو", "يوليو", "اغسطس", "أغسطس",
    "سبتمبر", "اكتوبر", "أكتوبر", "نوفمبر", "ديسمبر",
    "كانون", "شباط", "اذار", "نيسان", "ايار", "حزيران", "تموز", "ايلول", "تشرين",
})
NUMBER_WORDS = frozenset({
    "واحد", "اثنان", "اثنين", "ثلاث", "ثلاثة", "ثلاثه", "اربع", "اربعة", "خمس", "خمسة", "خمسه",
    "ست", "ستة", "سبع", "سبعة", "ثمان", "ثمانية", "ثمانيه", "تسع", "تسعة", "عشر", "عشرة",
    "عشرون", "عشرين", "ثلاثون", "ثلاثين", "مئة", "مائة",
    "اول", "ثاني", "ثالث", "رابع", "خامس", "سادس", "سابع", "ثامن", "تاسع", "عاشر",
})
SYMPTOM_WORDS = frozenset({"د", "دال", "اري", "أرى", "كلتا", "كلا", "امام", "أمام"})
KEEP_WORDS = NEGATION_WORDS | TIME_WORDS | BODY_WORDS | MONTH_WORDS | NUMBER_WORDS | SYMPTOM_WORDS


class CleanMode(str, Enum):
    NONE = "none"
    LIGHT = "light"
    HEAVY = "heavy"


@dataclass(frozen=True)
class CleanOptions:
    """كل الخيارات افتراضياً False، فتقدري تجرّبي خطوة وحدة: CleanOptions(remove_tatweel=True)."""
    strip_invisible: bool = False
    remove_diacritics: bool = False
    remove_tatweel: bool = False
    normalize_spaces: bool = False
    normalize_digits: bool = False
    lowercase: bool = False
    mask_numbers: bool = False
    collapse_repeats: bool = False
    normalize_alef: bool = False
    normalize_yeh: bool = False
    normalize_teh_marbuta: bool = False
    remove_punctuation: bool = False
    remove_stopwords: bool = False

    @classmethod
    def for_mode(cls, mode: CleanMode | str) -> "CleanOptions":
        mode = CleanMode(mode)
        if mode is CleanMode.NONE:
            return cls()
        if mode is CleanMode.LIGHT:
            return cls(strip_invisible=True, remove_diacritics=True, remove_tatweel=True, normalize_spaces=True)
        return cls(
            strip_invisible=True, remove_diacritics=True, remove_tatweel=True, normalize_spaces=True,
            normalize_digits=True, lowercase=True, mask_numbers=True, collapse_repeats=True,
            normalize_alef=True, normalize_yeh=True, normalize_teh_marbuta=True,
            remove_punctuation=True, remove_stopwords=True,
        )

    @classmethod
    def for_key(cls) -> "CleanOptions":
        """للمقارنة بين الأسئلة (كشف التكرار والتضارب): توحيد الكتابة بدون حذف كلمات ولا إخفاء أرقام."""
        return cls(
            strip_invisible=True, remove_diacritics=True, remove_tatweel=True, normalize_spaces=True,
            normalize_digits=True, lowercase=True, collapse_repeats=True,
            normalize_alef=True, normalize_yeh=True, normalize_teh_marbuta=True, remove_punctuation=True,
        )


class TextCleaner:
    def __init__(self, mode: CleanMode | str = CleanMode.LIGHT, options: CleanOptions | None = None):
        self.mode = CleanMode(mode)  # يقبل "light" أو CleanMode.LIGHT، ويرفض أي شي ثاني
        self.options = options or CleanOptions.for_mode(self.mode)
        self._stopwords = None

    def clean(self, text) -> str:
        """تتحمل NaN بدون ما تنهار. ترتيب الخطوات مقصود (الشرح بالدفتر)."""
        if pd.isna(text):
            return ""
        o = self.options
        t = str(text)
        if o.strip_invisible:
            t = _INVISIBLE.sub("", t)
        if o.remove_diacritics:
            t = _DIACRITICS.sub("", t)
        if o.remove_tatweel:
            t = t.replace(_TATWEEL, "")
        if o.normalize_digits:
            t = t.translate(_DIGITS)
        if o.lowercase:
            t = t.lower()
        if o.mask_numbers:
            t = _NUMBER.sub(f" {NUMBER_TOKEN} ", t)
        if o.collapse_repeats:
            t = _REPEATED_LETTER.sub(r"\1", t)
        if o.normalize_alef:
            t = t.translate(_ALEF)
        if o.normalize_yeh:
            t = t.translate(_YEH)
        if o.normalize_teh_marbuta:
            t = t.translate(_TEH_MARBUTA)
        if o.remove_punctuation:
            t = _PUNCTUATION.sub(" ", t)
        if o.normalize_spaces:
            t = _SPACES.sub(" ", t).strip()
        if o.remove_stopwords:
            stopwords = self.stopwords
            t = " ".join(word for word in t.split() if word not in stopwords)
        return t

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """ترجع نسخة من الجدول مع عمود clean_text."""
        out = df.copy()
        out[Columns.CLEAN] = out[Columns.TEXT].map(self.clean)
        return out

    @property
    def stopwords(self) -> frozenset:
        """قائمة NLTK العربية بعد حذف KEEP_WORDS منها، ومطبَّع عليها نفس تنظيف النص حتى تتطابق مع الكلمات."""
        if self._stopwords is None:
            self._stopwords = self._build_stopwords()
        return self._stopwords

    def _build_stopwords(self) -> frozenset:
        from nltk.corpus import stopwords

        try:
            raw_words = stopwords.words("arabic")
        except LookupError:
            import nltk

            nltk.download("stopwords", quiet=True)
            raw_words = stopwords.words("arabic")

        normalizer = TextCleaner(options=replace(self.options, remove_stopwords=False))
        keep = {normalizer.clean(word) for word in KEEP_WORDS}
        normalized = {normalizer.clean(word) for word in raw_words}
        return frozenset(word for word in normalized if word and word not in keep)


def clean_text(text, mode: CleanMode | str = "light") -> str:
    return TextCleaner(mode).clean(text)


def apply_clean(df: pd.DataFrame, mode: CleanMode | str = "light") -> pd.DataFrame:
    return TextCleaner(mode).transform(df)