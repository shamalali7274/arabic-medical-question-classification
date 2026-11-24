import pandas as pd
import pytest

from preprocess import CleanOptions, TextCleaner, apply_clean, clean_text


def test_none_mode_returns_text_unchanged():
    assert TextCleaner("none").clean("  مَرحباً  \n") == "  مَرحباً  \n"


def test_light_removes_diacritics_tatweel_and_extra_spaces():
    assert TextCleaner("light").clean("أَلَـــمٌ   في\nالبطن") == "ألم في البطن"


def test_light_keeps_punctuation_and_alef_forms():
    assert TextCleaner("light").clean("أين؟ إلى") == "أين؟ إلى"


def test_heavy_normalizes_letters_and_punctuation():
    out = TextCleaner("heavy").clean("عندي ألم، شديييد في المعدة؟!")
    assert out == "عندي الم شديد المعده"


def test_heavy_masks_numbers_including_arabic_indic_digits():
    out = TextCleaner("heavy").clean("عمري ٢٥ سنة والحرارة 38.5")
    assert out.split().count("NUM") == 2


def test_heavy_keeps_medical_codes_that_contain_digits():
    out = TextCleaner("heavy").clean("تحليل HbA1c وفيتامين B12 و 10mg يوميا و 6.5")
    assert "hba1c" in out.split() and "b12" in out.split() and "10mg" in out.split()
    assert out.split().count("NUM") == 1


def test_heavy_masks_digits_glued_to_arabic_words():
    out = TextCleaner("heavy").clean("عمري34 وبنتي ٨شهور وضغطي90/60").split()
    assert out.count("NUM") == 4 and "عمري" in out and "شهور" in out


def test_heavy_keeps_months_numbers_and_vitamin_letter():
    out = TextCleaner("heavy").clean("اخذت فيتامين دال في مارس وطفلي عمره تسع اشهر والحمل شهر خامس").split()
    for word in ("دال", "مارس", "تسع", "خامس"):
        assert word in out


def test_heavy_keeps_glued_negations():
    out = TextCleaner("heavy").clean("لا اعاني من حرارة ولا سعال وما عندي الم").split()
    assert "ولا" in out and "وما" in out


def test_heavy_masks_each_standalone_number_in_a_range():
    assert TextCleaner("heavy").clean("السكر 125-145 صباحا").split().count("NUM") == 2


def test_heavy_keeps_negation_words():
    out = TextCleaner("heavy").clean("لا يوجد الم بدون حرارة")
    assert "لا" in out.split() and "بدون" in out.split()


def test_heavy_keeps_medical_homonyms():
    out = TextCleaner("heavy").clean("ألم في عين نفس ضيق")
    assert "عين" in out.split() and "نفس" in out.split()


def test_heavy_removes_common_stopwords():
    assert "هل" not in TextCleaner("heavy").clean("هل هذا الدواء مناسب").split()


def test_punctuation_removal_covers_arabic_marks():
    out = TextCleaner(options=CleanOptions(remove_punctuation=True, normalize_spaces=True)).clean("صداع،دوخة؛ قيء؟")
    assert out == "صداع دوخة قيء"


def test_single_step_options_change_only_that_step():
    cleaner = TextCleaner(options=CleanOptions(normalize_alef=True))
    assert cleaner.clean("أإآ  ة") == "ااا  ة"


def test_survives_nan_and_non_string():
    cleaner = TextCleaner("heavy")
    assert cleaner.clean(float("nan")) == "" and cleaner.clean(None) == "" and cleaner.clean(123) == "NUM"


def test_transform_adds_column_and_keeps_rows():
    df = pd.DataFrame({"id": [1, 2], "question": ["سؤالٌ", None]})
    out = TextCleaner("light").transform(df)
    assert list(out["clean_text"]) == ["سؤال", ""] and len(out) == 2


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        TextCleaner("medium")