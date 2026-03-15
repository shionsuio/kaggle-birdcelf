# Repository Guidelines

## Project Overview
This repository is for BirdCLEF 2026 competition work. Keep the competition overview, rules summary, and any fixed background notes in tracked Markdown files so they stay easy to reference. If you already have the overview text prepared, copy it as-is into `README.md` or `docs/competition_overview.md` rather than rewriting it each time.

## Project Structure
Use the repository for code and experiment records only. Kaggle competition data must stay local and ignored by Git.

- `README.md`: repository entry point and competition summary
- `docs/SESSION_HANDOFF.md`: current state and restart checklist
- `experiments/`: one folder per experiment
- `src/`: shared training, feature, and inference utilities
- `tests/`: automated tests for reusable code
- local-only data: `train_audio/`, `train_soundscapes/`, `test_soundscapes/`, and top-level CSV files

## Experiment Workflow
Use one folder per experiment. Do not mix multiple ideas in the same directory.

Example:
```text
experiments/
  exp001_baseline/
    README.md
    train.py
    inference.py
    notes.md
```

Each experiment folder should record:
- hypothesis: what you expect to improve and why
- validation: how you tested it
- result: score, metric delta, and short interpretation
- next step: whether to keep, revise, or drop the idea

## Coding Style & Naming
Prefer Python with 4-space indentation and snake_case filenames such as `train_cnn.py` or `make_features.py`. Name experiment folders sequentially, for example `exp003_logmel_aug`. Keep shared logic in `src/`; keep experiment-specific code inside that experiment’s folder.

## Validation & Safety
Before committing, run `git status --short` and confirm that Kaggle audio, labels, and metadata are not staged. For reusable code, add `pytest` tests under `tests/` using `test_<feature>.py`. For experiment work, at minimum document the validation split, metric, and final score in the experiment’s `README.md`.

## Session Restart Workflow
If work is interrupted, restart from `docs/SESSION_HANDOFF.md`. Update that file whenever the repository structure changes materially, a new baseline becomes the default reference point, or the next recommended action changes.

## Commit & PR Guidelines
Write short imperative commits such as `Add exp004 conformer baseline` or `Document exp005 validation result`. Keep pull requests focused on one experiment or one shared-code change. In the PR description, link the experiment folder and summarize the hypothesis, validation setup, and result.
