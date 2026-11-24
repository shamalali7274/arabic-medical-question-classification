import pandas as pd

from quality import QualityFixer


def _train():
    return pd.DataFrame({
        "id": range(1, 11),
        "question": [
            "ألم في البطن منذ يومين",
            "ألم في البطن منذ يومين!",
            "عندي صداع شديد",
            "عندي صداع شديد",
            "   ",
            None,
            "هل الدواء مناسب",
            "هل الدواء مناسب",
            "طفلي عنده حرارة",
            "طفلي عنده حرارة عالية",
        ],
        "category": [
            "أمراض الأطفال", "امراض الاطفال", "الطب العام", "الطب العام", "الطب العام",
            "الطب العام", "الطب العام", "أمراض نسائية", "امراض الاطفال", "أمراض الأطفال",
        ],
    })


def test_label_groups_finds_spelling_variants():
    groups = QualityFixer().label_groups(_train())
    assert set(groups["label"]) == {"أمراض الأطفال", "امراض الاطفال"}


def test_unify_labels_uses_most_frequent_variant_when_no_names_given():
    fixed = QualityFixer().apply(_train(), is_train=True)
    assert fixed["category"].isin(["أمراض الأطفال", "امراض الاطفال"]).sum() == fixed["category"].str.contains("الاطفال|الأطفال").sum()
    assert fixed.loc[fixed["category"].str.contains("طفا|طفال"), "category"].nunique() == 1


def test_unify_labels_uses_official_names_when_given():
    fixed = QualityFixer(label_names=["أمراض الأطفال", "الطب العام", "أمراض نسائية"]).apply(_train(), is_train=True)
    assert "امراض الاطفال" not in set(fixed["category"])


def test_drops_empty_and_nan_questions():
    fixed = QualityFixer().apply(_train(), is_train=True)
    assert fixed["question"].str.strip().ne("").all() and fixed["question"].notna().all()


def test_drops_duplicates_after_normalization_keeping_first():
    fixed = QualityFixer().apply(_train(), is_train=True)
    assert (fixed["question"] == "ألم في البطن منذ يومين").sum() == 1
    assert (fixed["question"] == "عندي صداع شديد").sum() == 1


def test_drops_conflicting_labels_entirely():
    fixed = QualityFixer().apply(_train(), is_train=True)
    assert "هل الدواء مناسب" not in set(fixed["question"])


def test_conflicts_can_be_kept():
    fixed = QualityFixer(drop_conflicts=False).apply(_train(), is_train=True)
    assert (fixed["question"] == "هل الدواء مناسب").sum() == 2


def test_test_split_never_loses_rows_and_nan_becomes_empty():
    test = pd.DataFrame({"id": [1, 2, 3], "question": ["سؤال", None, "سؤال"]})
    fixed = QualityFixer().apply(test, is_train=False)
    assert len(fixed) == 3 and fixed["question"].tolist() == ["سؤال", "", "سؤال"]


def test_leakage_removed_only_when_requested_and_reference_given():
    train = pd.DataFrame({"id": [1], "question": ["عندي صداع"], "category": ["الطب العام"]})
    val = pd.DataFrame({"id": [2, 3], "question": ["عندي صداع!", "ألم في العين"], "category": ["الطب العام", "امراض العيون"]})
    kept = QualityFixer().apply(val, is_train=True, reference=train)
    dropped = QualityFixer(drop_leaked=True).apply(val, is_train=True, reference=train)
    assert len(kept) == 2 and dropped["id"].tolist() == [3]


def test_low_arabic_ratio_rows_dropped_when_enabled():
    df = pd.DataFrame({"id": [1, 2], "question": ["I have a headache", "عندي صداع"], "category": ["الطب العام"] * 2})
    fixed = QualityFixer(min_arabic_ratio=0.5).apply(df, is_train=True)
    assert fixed["id"].tolist() == [2]


def test_summary_reports_rows_removed_per_step():
    fixer = QualityFixer()
    fixer.apply(_train(), is_train=True)
    table = fixer.summary()
    assert table["step"].iloc[0] == "البداية" and table["removed"].sum() == 10 - table["rows"].iloc[-1]


def test_is_train_false_unifies_labels_without_dropping_rows():
    val = pd.DataFrame({
        "id": [1, 2, 3],
        "question": ["عندي صداع", "عندي صداع", None],
        "category": ["الطب العام", "امراض الاطفال", "الطب العام"],
    })
    fixed = QualityFixer(label_names=["الطب العام", "أمراض الأطفال"]).apply(val, is_train=False)
    assert len(fixed) == 3
    assert fixed["category"].tolist() == ["الطب العام", "أمراض الأطفال", "الطب العام"]