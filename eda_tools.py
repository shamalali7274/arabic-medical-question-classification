"""EDA toolkit (Person 1). Every analysis in ``01_eda.ipynb`` is a small class here,
so the notebook stays readable and each piece can be unit-tested.

Rules respected by this module:
* it never modifies the input DataFrames,
* the TF-IDF vectorizer is fitted on TRAIN only,
* nothing here deletes rows from test.

``ArabicText.normalize`` is an *analysis-only* normalizer used to compare strings
(duplicates, label spellings). The real cleaning pipeline is ``preprocess.py`` (Person 2).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from common import Columns

# --------------------------------------------------------------------------- text helpers


class ArabicText:
    """Light Arabic normalization used only to *compare* strings during analysis."""

    DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
    TATWEEL = "\u0640"
    ALEF_FORMS = re.compile(r"[أإآٱ]")
    INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    WORD = re.compile(r"[\u0621-\u064A]+|[A-Za-z]+|\d+")

    @classmethod
    def normalize(cls, text) -> str:
        if pd.isna(text):
            return ""
        text = str(text).translate(cls.INDIC_DIGITS)
        text = cls.DIACRITICS.sub("", text).replace(cls.TATWEEL, "")
        text = cls.ALEF_FORMS.sub("ا", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    @classmethod
    def tokens(cls, text) -> list[str]:
        return cls.WORD.findall(cls.normalize(text))

    @classmethod
    def normalize_series(cls, series: pd.Series) -> pd.Series:
        return series.map(cls.normalize)


# --------------------------------------------------------------------------- overview


class SplitOverview:
    """Sizes, dtypes, nulls and id hygiene for every split."""

    def __init__(self, splits: Mapping[str, pd.DataFrame]) -> None:
        self.splits = dict(splits)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, df in self.splits.items():
            text = df[Columns.TEXT]
            rows.append({
                "split": name,
                "rows": len(df),
                "columns": ", ".join(df.columns),
                "null_cells": int(df.isna().sum().sum()),
                "empty_text": int((text.fillna("").str.strip() == "").sum()),
                "unique_ids": df[Columns.ID].nunique(),
                "labels": df[Columns.LABEL].nunique() if Columns.LABEL in df else None,
            })
        return pd.DataFrame(rows).set_index("split")

    def id_overlap(self) -> pd.DataFrame:
        """How many ids are shared between each pair of splits (should be 0)."""
        names = list(self.splits)
        ids = {n: set(self.splits[n][Columns.ID]) for n in names}
        rows = [{"pair": f"{a} & {b}", "shared_ids": len(ids[a] & ids[b])}
                for i, a in enumerate(names) for b in names[i + 1:]]
        return pd.DataFrame(rows).set_index("pair")


# --------------------------------------------------------------------------- labels


class LabelAuditor:
    """Compare label spellings in train/validation against ``label_mapping``."""

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, label_map: pd.DataFrame) -> None:
        self.train, self.val, self.label_map = train, val, label_map

    def audit(self) -> pd.DataFrame:
        """One row per raw label string, with counts and where it matches the mapping."""
        exact = set(self.label_map[Columns.LABEL])
        mapped_keys = {ArabicText.normalize(c) for c in exact}
        table = pd.DataFrame({
            "train": self.train[Columns.LABEL].value_counts(),
            "val": self.val[Columns.LABEL].value_counts(),
        }).fillna(0).astype(int)
        table["in_mapping_exactly"] = [label in exact for label in table.index]
        table["key"] = [ArabicText.normalize(label) for label in table.index]
        table["key_in_mapping"] = table["key"].isin(mapped_keys)
        table["has_edge_spaces"] = [label != label.strip() for label in table.index]
        return table.sort_values(["key", "train"], ascending=[True, False])

    def variant_groups(self) -> pd.DataFrame:
        """Raw spellings that collapse to the same normalized label."""
        audit = self.audit()
        groups = audit.groupby("key").filter(lambda g: len(g) > 1)
        return groups[["key", "train", "val"]]

    def canonical_label_count(self) -> int:
        return self.audit()["key"].nunique()

    def labels_missing_in_val(self) -> list[str]:
        audit = self.audit()
        return audit.index[(audit["train"] > 0) & (audit["val"] == 0)].tolist()


class ClassDistribution:
    """Counts and percentages per class, plus imbalance measures."""

    @staticmethod
    def table(df: pd.DataFrame, by: pd.Series | None = None) -> pd.DataFrame:
        labels = df[Columns.LABEL] if by is None else by
        counts = labels.value_counts()
        return pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(2)})

    @classmethod
    def compare(cls, train: pd.DataFrame, val: pd.DataFrame, by_key: bool = False) -> pd.DataFrame:
        """Side-by-side distribution; ``by_key`` merges spelling variants first."""
        f = (lambda d: d[Columns.LABEL].map(ArabicText.normalize)) if by_key else (lambda d: d[Columns.LABEL])
        left, right = cls.table(train, f(train)), cls.table(val, f(val))
        out = left.join(right, lsuffix="_train", rsuffix="_val", how="outer").fillna(0)
        out["abs_diff_pct"] = (out["percent_train"] - out["percent_val"]).abs().round(2)
        return out.sort_values("count_train", ascending=False)

    @staticmethod
    def imbalance_ratio(counts: pd.Series) -> float:
        return float(counts.max() / counts.min())

    @staticmethod
    def uniform_share(n_classes: int) -> float:
        return round(100 / n_classes, 2)


# --------------------------------------------------------------------------- lengths & words


class LengthProfile:
    """Word and character counts, overall and per class."""

    WORDS, CHARS = "n_words", "n_chars"

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        text = out[Columns.TEXT].fillna("")
        out[self.WORDS] = text.str.split().str.len()
        out[self.CHARS] = text.str.len()
        return out

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        with_len = self.add(df)
        return with_len[[self.WORDS, self.CHARS]].describe(percentiles=[.05, .25, .5, .75, .95]).round(1)

    def by_class(self, df: pd.DataFrame, key: pd.Series | None = None) -> pd.DataFrame:
        with_len = self.add(df)
        group = with_len[Columns.LABEL] if key is None else key
        return (with_len.groupby(group)[self.WORDS]
                .agg(["count", "mean", "median", "min", "max"]).round(1)
                .sort_values("median", ascending=False))

    def extremes(self, df: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
        with_len = self.add(df)[[Columns.ID, Columns.TEXT, self.WORDS, Columns.LABEL]]
        ordered = with_len.sort_values(self.WORDS)
        return ordered.head(n), ordered.tail(n).iloc[::-1]


class WordFrequency:
    """Most frequent words, overall and per class, and the words shared by many classes.

    ``normalize=True`` tokenizes the normalized text (letters only, hamza variants merged);
    ``normalize=False`` is the raw whitespace split, used for the before/after comparison.
    """

    def __init__(self, min_len: int = 2, normalize: bool = True) -> None:
        self.min_len = min_len
        self.normalize = normalize

    def _tokens(self, text) -> list[str]:
        if self.normalize:
            return ArabicText.tokens(text)
        return [] if pd.isna(text) else str(text).split()

    def _counter(self, texts: Iterable[str]) -> Counter:
        counter: Counter = Counter()
        for text in texts:
            counter.update(t for t in self._tokens(text) if len(t) >= self.min_len)
        return counter

    def vocabulary_size(self, df: pd.DataFrame) -> int:
        return len(self._counter(df[Columns.TEXT]))

    def top_words(self, df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
        counter = self._counter(df[Columns.TEXT])
        total = sum(counter.values())
        rows = [(w, c, round(c / total * 100, 2)) for w, c in counter.most_common(n)]
        return pd.DataFrame(rows, columns=["word", "count", "percent_of_tokens"])

    def top_words_by_class(self, df: pd.DataFrame, key: pd.Series, n: int = 10) -> pd.DataFrame:
        columns = {}
        for label, group in df.groupby(key):
            columns[label] = [w for w, _ in self._counter(group[Columns.TEXT]).most_common(n)]
        return pd.DataFrame(columns)

    def shared_top_words(self, df: pd.DataFrame, key: pd.Series, n: int = 30, min_classes: int = 8) -> pd.DataFrame:
        """Words that appear in the top-n of at least ``min_classes`` classes (generic words)."""
        presence: Counter = Counter()
        for _, group in df.groupby(key):
            presence.update(w for w, _ in self._counter(group[Columns.TEXT]).most_common(n))
        rows = [(w, k) for w, k in presence.items() if k >= min_classes]
        return (pd.DataFrame(rows, columns=["word", "classes_in_top_n"])
                .sort_values("classes_in_top_n", ascending=False).reset_index(drop=True))


class PatternCounter:
    """Count how many questions contain each text pattern (regex), with an example."""

    DEFAULT_PATTERNS: dict[str, str] = {
        "Arabic-Indic digits (٠-٩)": r"[٠-٩]",
        "Western digits (0-9)": r"[0-9]",
        "Latin letters": r"[A-Za-z]",
        "Emoji": r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
        "URL": r"https?://|www\.",
        "Diacritics (tashkeel)": r"[\u064B-\u065F\u0670]",
        "Tatweel (ـ)": r"\u0640",
        "Repeated letters (3+)": r"(.)\1{2,}",
        "Greeting": r"السلام عليكم|سلام عليكم|مرحبا|اهلا|أهلا|صباح الخير|مساء الخير",
        "Thanks / dua": r"شكرا|شكراً|جزاكم|بارك الله|يعطيكم العافية|الله يعافيكم",
        "Doctor address": r"دكتور|دكتوره|دكتورة|طبيب|د\.",
        "Alef with hamza (أ إ آ)": r"[أإآ]",
        "Alef maqsura (ى)": r"ى",
        "Yeh (ي)": r"ي",
        "Teh marbuta (ة)": r"ة",
        "Heh (ه)": r"ه",
        "Dialect words": r"\b(شو|ليش|هاي|هاد|هيك|بدي|وين|ايش|إيش|مو|عشان|شلون|انو|أنو|كتير|لسا|دلوقتي|ازاي|عايز|عندي)\b",
        "Arabic punctuation (، ؛ ؟)": r"[،؛؟]",
        "Latin punctuation (. , ? ! :)": r"[.,?!:]",
        "Newline": r"[\n\r]",
        "Double spaces": r" {2,}",
    }

    def __init__(self, patterns: Mapping[str, str] | None = None) -> None:
        self.patterns = {k: re.compile(v) for k, v in (patterns or self.DEFAULT_PATTERNS).items()}

    def count(self, df: pd.DataFrame) -> pd.DataFrame:
        text = df[Columns.TEXT].fillna("")
        rows = []
        for name, rx in self.patterns.items():
            mask = text.map(lambda t: rx.search(t) is not None)
            example = text[mask].iloc[0] if mask.any() else ""
            rows.append({
                "pattern": name,
                "rows": int(mask.sum()),
                "percent": round(mask.mean() * 100, 2),
                "example": example[:90],
            })
        return pd.DataFrame(rows).set_index("pattern")


# --------------------------------------------------------------------------- duplicates, similarity, leakage


class DuplicateFinder:
    """Exact duplicates, within a split and between splits (raw or normalized text)."""

    @staticmethod
    def _key(df: pd.DataFrame, normalized: bool) -> pd.Series:
        text = df[Columns.TEXT].fillna("")
        return ArabicText.normalize_series(text) if normalized else text.astype(str)

    def within(self, df: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
        """Every row that belongs to a duplicated group (including the first copy)."""
        key = self._key(df, normalized)
        dup = df[key.duplicated(keep=False)].copy()
        dup["_key"] = key[dup.index]
        return dup.sort_values("_key")

    def label_conflicts(self, df: pd.DataFrame, normalized: bool = True) -> pd.DataFrame:
        """Duplicated questions that carry more than one (normalized) label."""
        dup = self.within(df, normalized)
        if dup.empty:
            return dup
        dup["_label"] = dup[Columns.LABEL].map(ArabicText.normalize)
        mixed = dup.groupby("_key")["_label"].transform("nunique") > 1
        return dup[mixed]

    def between(self, left: pd.DataFrame, right: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
        """Rows of ``left`` whose text also appears in ``right``."""
        right_keys = set(self._key(right, normalized))
        return left[self._key(left, normalized).isin(right_keys)]


class SimilarityFinder:
    """Near-duplicate search with TF-IDF + cosine nearest neighbours, in batches.

    The vectorizer is fitted on ``reference`` (train) only.
    """

    def __init__(self, reference: pd.DataFrame, batch_size: int = 2000, n_jobs: int = -1) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        self.reference = reference.reset_index(drop=True)
        self.batch_size = batch_size
        self._vectorizer = TfidfVectorizer(
            tokenizer=ArabicText.tokens, preprocessor=None, lowercase=False,
            token_pattern=None, sublinear_tf=True, min_df=1, dtype=np.float32,
        )
        self._matrix = self._vectorizer.fit_transform(self.reference[Columns.TEXT].fillna(""))
        self._index = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=n_jobs)
        self._index.fit(self._matrix)

    def _query(self, matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, positions = [], []
        for start in range(0, matrix.shape[0], self.batch_size):
            d, p = self._index.kneighbors(matrix[start:start + self.batch_size], n_neighbors=k)
            distances.append(d)
            positions.append(p)
        return np.vstack(distances), np.vstack(positions)

    def nearest_in_reference(self, queries: pd.DataFrame) -> pd.DataFrame:
        """For every query row: its most similar row in the reference set."""
        matrix = self._vectorizer.transform(queries[Columns.TEXT].fillna(""))
        dist, pos = self._query(matrix, k=1)
        out = pd.DataFrame({
            "query_id": queries[Columns.ID].to_numpy(),
            "query_text": queries[Columns.TEXT].to_numpy(),
            "match_id": self.reference[Columns.ID].to_numpy()[pos[:, 0]],
            "match_text": self.reference[Columns.TEXT].to_numpy()[pos[:, 0]],
            "similarity": (1 - dist[:, 0]).round(4),
        })
        if Columns.LABEL in queries and Columns.LABEL in self.reference:
            out["query_label"] = queries[Columns.LABEL].to_numpy()
            out["match_label"] = self.reference[Columns.LABEL].to_numpy()[pos[:, 0]]
        return out

    def nearest_within_reference(self) -> pd.DataFrame:
        """For every reference row: its most similar OTHER row in the same set."""
        dist, pos = self._query(self._matrix, k=2)
        # column 0 is the row itself unless an exact duplicate sorts first; pick the first different position
        own = np.arange(len(self.reference))
        neighbour = np.where(pos[:, 0] == own, pos[:, 1], pos[:, 0])
        distance = np.where(pos[:, 0] == own, dist[:, 1], dist[:, 0])
        out = pd.DataFrame({
            "id": self.reference[Columns.ID].to_numpy(),
            "text": self.reference[Columns.TEXT].to_numpy(),
            "neighbour_id": self.reference[Columns.ID].to_numpy()[neighbour],
            "neighbour_text": self.reference[Columns.TEXT].to_numpy()[neighbour],
            "similarity": (1 - distance).round(4),
        })
        if Columns.LABEL in self.reference:
            out["label"] = self.reference[Columns.LABEL].to_numpy()
            out["neighbour_label"] = self.reference[Columns.LABEL].to_numpy()[neighbour]
        return out

    @staticmethod
    def threshold_table(similarity: pd.Series, thresholds: Iterable[float] = (0.99, 0.95, 0.90, 0.80, 0.70)) -> pd.DataFrame:
        rows = [{"similarity >=": t, "rows": int((similarity >= t).sum()),
                 "percent": round((similarity >= t).mean() * 100, 2)} for t in thresholds]
        return pd.DataFrame(rows).set_index("similarity >=")


@dataclass
class LeakageRow:
    pair: str
    rows_checked: int
    exact_raw: int
    exact_normalized: int

    @property
    def percent_normalized(self) -> float:
        return round(self.exact_normalized / self.rows_checked * 100, 2)


class LeakageDetector:
    """Questions that appear in more than one split (train/validation/test)."""

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        self.train, self.val, self.test = train, val, test
        self._finder = DuplicateFinder()

    def exact_report(self) -> pd.DataFrame:
        pairs = [("validation in train", self.val, self.train),
                 ("test in train", self.test, self.train),
                 ("test in validation", self.test, self.val)]
        rows = []
        for name, left, right in pairs:
            rows.append(LeakageRow(
                pair=name, rows_checked=len(left),
                exact_raw=len(self._finder.between(left, right, normalized=False)),
                exact_normalized=len(self._finder.between(left, right, normalized=True)),
            ))
        out = pd.DataFrame([r.__dict__ for r in rows]).set_index("pair")
        out["percent_normalized"] = [r.percent_normalized for r in rows]
        return out

    def val_rows_matching_train(self) -> pd.DataFrame:
        """Validation rows whose normalized text is in train, with the train label(s) found."""
        train_key = ArabicText.normalize_series(self.train[Columns.TEXT].fillna(""))
        train_label = self.train[Columns.LABEL].map(ArabicText.normalize)
        lookup = train_label.groupby(train_key).agg(lambda s: set(s))
        val = self.val.copy()
        val["_key"] = ArabicText.normalize_series(val[Columns.TEXT].fillna(""))
        leaked = val[val["_key"].isin(lookup.index)].copy()
        leaked["train_labels"] = leaked["_key"].map(lookup)
        leaked["same_label"] = [ArabicText.normalize(lbl) in labels
                                for lbl, labels in zip(leaked[Columns.LABEL], leaked["train_labels"])]
        return leaked

    def test_rows_matching_train(self) -> pd.DataFrame:
        """Informational only: test rows are never dropped."""
        return self._finder.between(self.test, self.train, normalized=True)

    def val_without_leakage(self) -> pd.DataFrame:
        """Validation minus the rows whose normalized text also appears in train."""
        leaked_index = self.val_rows_matching_train().index
        return self.val.drop(index=leaked_index)


# --------------------------------------------------------------------------- plotting


class ArabicPlotter:
    """Matplotlib helpers that shape Arabic text correctly (falls back gracefully)."""

    def __init__(self) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.style.use("ggplot")
        plt.rcParams["font.sans-serif"] = ["Arial", "Tahoma", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        self._shape = self._build_shaper()

    @staticmethod
    def _build_shaper():
        """matplotlib >= 3.11 bundles libraqm and shapes Arabic itself; older versions need reshaper + bidi."""
        from matplotlib import ft2font

        if hasattr(ft2font, "__libraqm_version__"):
            return str
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
        except ImportError:  # plots still work, Arabic just renders unshaped
            return str
        return lambda s: get_display(arabic_reshaper.reshape(str(s)))

    def ar(self, text) -> str:
        return self._shape(text)

    def barh_counts(self, counts: pd.Series, title: str, xlabel: str = "count", ax=None):
        ax = ax or self.plt.subplots(figsize=(9, 6))[1]
        ordered = counts.sort_values()
        ax.barh([self.ar(i) for i in ordered.index], ordered.values)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        return ax

    def grouped_barh(self, table: pd.DataFrame, columns: list[str], title: str, xlabel: str = "percent"):
        fig, ax = self.plt.subplots(figsize=(10, 7))
        ordered = table.sort_values(columns[0])
        y = np.arange(len(ordered))
        height = 0.8 / len(columns)
        for i, col in enumerate(columns):
            ax.barh(y + i * height, ordered[col], height=height, label=col)
        ax.set_yticks(y + height * (len(columns) - 1) / 2)
        ax.set_yticklabels([self.ar(i) for i in ordered.index])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return ax

    def histogram(self, values: pd.Series, title: str, xlabel: str, bins: int = 60, clip: float | None = None):
        fig, ax = self.plt.subplots(figsize=(9, 4))
        data = values.clip(upper=clip) if clip else values
        ax.hist(data, bins=bins)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("questions")
        fig.tight_layout()
        return ax

    def boxplot_by_class(self, df: pd.DataFrame, value: str, key: pd.Series, title: str, clip: float | None = None):
        fig, ax = self.plt.subplots(figsize=(10, 6))
        order = df.groupby(key)[value].median().sort_values().index
        data = [df.loc[key == label, value].clip(upper=clip) if clip else df.loc[key == label, value] for label in order]
        ax.boxplot(data, vert=False, showfliers=False)
        ax.set_yticklabels([self.ar(label) for label in order])
        ax.set_xlabel(value)
        ax.set_title(title)
        fig.tight_layout()
        return ax

    def barh_values(self, labels: Iterable[str], values: Iterable[float], title: str, xlabel: str):
        fig, ax = self.plt.subplots(figsize=(9, 6))
        labels, values = list(labels), list(values)
        ax.barh([self.ar(label) for label in labels][::-1], values[::-1])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        return ax
