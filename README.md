# Arabic Medical Question Classification (NLP practical, part 1)

Classify Arabic medical questions (altibbi.com) into 10 specialties. This repo holds the shared
code contract and Person 1's work (structure, EDA, similarity/leakage, merge and submission).

## Layout
| File | Purpose | Owner |
|---|---|---|
| `common.py` | `DATA_DIR`, `RANDOM_STATE`, `Columns`, `load_data()` (OOP: `DataLoader`, `Dataset`) | P1 |
| `quality.py` | `apply_quality_fixes(df, is_train)` (placeholder) | P2 |
| `preprocess.py` | `clean_text(text, mode)`, `apply_clean(df, mode)` (placeholder) | P2 |
| `eda_tools.py` | Analysis classes used by `01_eda.ipynb` | P1 |
| `submission.py` | `SubmissionBuilder`: builds and validates `submission.csv` | P1 |
| `notebooks/01_eda.ipynb` | EDA, similarity, leakage | P1 |
| `tests/` | `pytest` suite (23 tests) | P1 |

## Setup
```bash
pip install -r requirements.txt
# put the CSV files in data/  (or set MEDQ_DATA_DIR=<folder>; on Kaggle it is found automatically)
pytest -q
jupyter notebook notebooks/01_eda.ipynb
```
`load_data()` accepts `test.csv` or `test 1.csv`, and `label_mapping.csv` / `labels_mapping.csv` / `labels_mapping.xlsx`.

## Team rules (from the contract)
* Nobody renames a function or column without telling the others. Columns: `id, question, category, clean_text`.
* Never delete a row from test. Deleting/unifying is allowed on train and validation only.
* Fit the vectorizer on train only.
* One branch and one notebook each (`p1-eda`/01, `p2-quality`/02, `p3-baseline`/03). Clear notebook outputs before committing.
* Never commit data files or keys (`data/`, `*.csv`, `*.xlsx` are git-ignored).

## Producing the submission (Wednesday)
```python
from common import DataLoader
from submission import SubmissionBuilder

loader = DataLoader()
_, _, test, label_map = loader.load().as_tuple()
builder = SubmissionBuilder(loader.load_sample_submission(), label_map)
submission = builder.build(test, predictions)      # predictions: names or label ids, in test row order
builder.save(submission, "submission.csv")          # raises SubmissionError if the format is wrong
```
