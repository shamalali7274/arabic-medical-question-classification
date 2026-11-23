import pandas as pd

from common import Columns
from eda_tools import (ArabicText, ClassDistribution, DuplicateFinder, LabelAuditor, LeakageDetector,
                       LengthProfile, PatternCounter, SimilarityFinder, SplitOverview, WordFrequency)


def _df(texts, labels=None, start=0):
    data = {"id": range(start, start + len(texts)), "question": texts}
    if labels is not None:
        data["category"] = labels
    return pd.DataFrame(data)


def test_normalize_unifies_alef_digits_diacritics_and_spaces():
    assert ArabicText.normalize("أَمراض  الأطفال ٢") == "امراض الاطفال 2"
    assert ArabicText.normalize(None) == ""


def test_label_auditor_finds_spelling_variants():
    train = _df(["a", "b", "c"], ["أمراض الدم", "امراض الدم", "الطب العام"])
    val = _df(["d"], ["أمراض الدم"], 10)
    label_map = pd.DataFrame({"category": ["أمراض الدم", "الطب العام"], "label_id": [0, 1]})
    auditor = LabelAuditor(train, val, label_map)
    assert auditor.canonical_label_count() == 2
    assert len(auditor.variant_groups()) == 2
    assert auditor.audit().loc["امراض الدم", "in_mapping_exactly"] == False  # noqa: E712


def test_class_distribution_merges_variants():
    train = _df(["a"] * 4, ["أمراض الدم", "امراض الدم", "الطب", "الطب"])
    out = ClassDistribution.compare(train, train, by_key=True)
    assert out.loc["امراض الدم", "count_train"] == 2 and ClassDistribution.imbalance_ratio(out["count_train"]) == 1


def test_split_overview_counts_empty_and_shared_ids():
    a, b = _df(["x", " "], ["l", "l"]), _df(["y"], ["l"], start=1)
    ov = SplitOverview({"a": a, "b": b})
    assert ov.summary().loc["a", "empty_text"] == 1
    assert ov.id_overlap().loc["a & b", "shared_ids"] == 1


def test_length_profile_and_pattern_counter():
    df = _df(["سؤال واحد", "ثلاث كلمات هنا ٣"], ["l", "l"])
    assert LengthProfile().add(df)["n_words"].tolist() == [2, 4]
    counts = PatternCounter().count(df)
    assert counts.loc["Arabic-Indic digits (٠-٩)", "rows"] == 1


def test_word_frequency_normalized_vs_raw_vocab():
    df = _df(["أعاني من ألم.", "اعاني من الم"], ["l", "l"])
    assert WordFrequency().vocabulary_size(df) < WordFrequency(normalize=False).vocabulary_size(df)


def test_duplicate_finder_normalized_catches_more_than_raw():
    df = _df(["أنا مريض", "انا مريض", "شي ثاني"], ["x", "y", "x"])
    finder = DuplicateFinder()
    assert len(finder.within(df, normalized=False)) == 0
    assert len(finder.within(df, normalized=True)) == 2
    assert len(finder.label_conflicts(df)) == 2


def test_leakage_detector_reports_and_removes_only_from_val():
    train = _df(["سؤال أ", "سؤال ب"], ["l1", "l2"])
    val = _df(["سؤال أ", "سؤال ج"], ["l1", "l2"], 10)
    test = _df(["سؤال ب", "سؤال د"], start=20)
    leak = LeakageDetector(train, val, test)
    report = leak.exact_report()
    assert report.loc["validation in train", "exact_normalized"] == 1
    assert report.loc["test in train", "exact_normalized"] == 1
    assert len(leak.val_without_leakage()) == 1
    assert len(leak.test) == 2  # test is never reduced


def test_similarity_finder_ranks_identical_text_first():
    train = _df(["ألم شديد في الظهر", "تساقط الشعر المستمر", "ضيق في التنفس ليلا"], ["a", "b", "c"])
    queries = _df(["ألم شديد في الظهر", "ضيق تنفس"], ["a", "c"], 100)
    out = SimilarityFinder(train, batch_size=1, n_jobs=1).nearest_in_reference(queries)
    assert out.loc[0, "similarity"] > 0.99 and out.loc[1, "match_label"] == "c"
