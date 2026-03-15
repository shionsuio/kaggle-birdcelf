# Repository Guidelines

## Project Structure & Module Organization
This repository should store code, notes, and lightweight metadata only. Kaggle competition files currently live at the top level (`train.csv`, `taxonomy.csv`, `train_soundscapes_labels.csv`, `sample_submission.csv`) and under `train_audio/`, `train_soundscapes/`, and `test_soundscapes/`, but they must remain ignored and must not be pushed to GitHub. If you add code later, prefer `src/` for reusable modules and `tests/` for automated checks.

## Build, Test, and Development Commands
There is no build system in this snapshot. Use quick validation commands locally before submitting changes:
- `python -m csv.tool train.csv | head` checks CSV readability.
- `find train_audio -type f | wc -l` verifies clip counts after data updates.
- `find train_soundscapes -type f | wc -l` confirms soundscape inventory.
- `git status --short` confirms that competition data is still ignored before commit.
- `python - <<'PY' ... PY` scripts are acceptable for one-off audits, but keep reusable logic in a tracked script if you add code later.

## Coding Style & Naming Conventions
If you add scripts or notebooks, prefer Python with 4-space indentation, snake_case filenames, and descriptive function names such as `validate_soundscape_labels.py`. Keep derived artifacts out of the repository unless they are required inputs. Preserve existing label keys like `primary_label` and directory names under `train_audio/`; downstream training pipelines usually depend on them exactly.

## Testing Guidelines
For data changes, test integrity rather than unit behavior. Validate that CSV headers remain unchanged, referenced audio files exist, and no duplicate filenames are introduced. If you add code, place tests in `tests/` and name them `test_<feature>.py` so they run cleanly with `pytest`.

## Commit & Pull Request Guidelines
The current history is minimal, so use short, imperative commit subjects such as `Add dataset ignore rules` or `Fix mislabeled soundscape rows`. In pull requests, list changed files, note any record-count deltas, describe validation commands run, and include small tables or screenshots only when they clarify label or taxonomy changes.

## Data Handling Notes
Do not commit or publish Kaggle competition audio, labels, or metadata to this repository. Keep raw files local, document preprocessing steps instead of uploading outputs, and use `.gitignore` to block accidental `git add .` commits. Do not overwrite raw audio in place without documenting the source and reason.
