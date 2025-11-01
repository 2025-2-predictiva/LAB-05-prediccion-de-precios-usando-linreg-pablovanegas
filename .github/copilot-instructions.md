## Purpose

Short, actionable guidance for AI coding agents working on this repository (a lab for predicting used car prices using linear regression).

## Quick start (developer workflow)

- Create and activate a Python venv (Windows example used by graders):

  ```cmd
  python3 -m venv .venv
  .venv\Scripts\activate
  setup
  ```

- Run tests with:

  ```cmd
  pytest
  ```

Dependencies are listed in `requirements.txt` (pandas, scikit-learn, pytest).

## Key files to read and update

- `homework/homework.py`: main student task and canonical description of required pipeline steps (Age column, drop Year and Car_Name, build pipeline, cross-validate, save model and metrics).
- `tests/test_homework.py`: autograder — contains the exact checks your changes must satisfy (component names, model file path, metrics format and thresholds). Read this file before editing code.
- `files/input/`: input datasets (train/test split provided for students).
- `files/grading/`: pickled grading datasets used by tests (x_train, y_train, x_test, y_test).
- `files/models/model.pkl.gz`: model output path expected by tests.
- `files/output/metrics.json`: metrics output file — tests read one JSON object per line.

## Project-specific expectations and patterns (do not change tests)

- Pipeline composition: the grader expects a pipeline containing these components (string names checked in `tests/test_homework.py`):
  - `OneHotEncoder`
  - `MinMaxScaler`
  - `SelectKBest`
  - `LinearRegression`

  The model stored must be a `GridSearchCV` object wrapping the pipeline (the test asserts that the loaded model's type string contains `GridSearchCV`).

- Model saving: compress the model with gzip and save to `files/models/model.pkl.gz` (tests assert the file exists and can be loaded with gzip + pickle).

- Metrics output format: write one JSON object per line to `files/output/metrics.json`. Each JSON object must be like:

  ```json
  {"type": "metrics", "dataset": "train", "r2": <float>, "mse": <float>, "mad": <float>}
  ```

- Data preprocessing specifics (from `homework/homework.py` instructions):
  - Create column `Age` as `2021 - Year`.
  - Drop `Year` and `Car_Name`.

## Tests & scoring notes (what the autograder checks)

- File presence: `files/models/model.pkl.gz` and `files/output/metrics.json` must exist.
- The pickled model must be loadable via gzip + pickle and be a `GridSearchCV` instance.
- The pipeline must include the component names listed above (tests search the string representation of pipeline components).
- Numeric thresholds: tests compare produced train/test metrics against fixed reference numbers — ensure your model performs at least as well as required in `tests/test_homework.py` (they assert comparisons like `r2 > 0.889` for train, etc.). See that file for exact numbers.

## Useful examples and snippets

- Loading the grading data for local experiments:

  - `files/grading/x_train.pkl`, `y_train.pkl`, `x_test.pkl`, `y_test.pkl` are pickles used by tests; you can load them with `pickle` or `pandas`.

- Save a gzip-compressed pickled model example (Python):

  ```py
  import gzip, pickle
  with gzip.open('files/models/model.pkl.gz', 'wb') as f:
      pickle.dump(grid_search_obj, f)
  ```

## What to avoid

- Do not change `tests/test_homework.py` — it codifies the autograder behavior.
- Do not change file paths or output formats expected by the tests (model path, metrics JSON lines).

## When to ask for clarification

- If a requested change would modify any of the files under `tests/` or the expected file locations, ask before proceeding — graders depend on those exact names/structures.

---

If any part of this guidance is unclear or you want more examples (e.g., a minimal pipeline that satisfies the tests), tell me which section to expand and I will iterate.
