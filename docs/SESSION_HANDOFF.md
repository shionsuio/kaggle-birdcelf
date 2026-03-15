# Session Handoff

## Current Repository Rules
- Kaggle competition data must stay local and must not be committed
- Use one folder per experiment under `experiments/`
- Each experiment should record hypothesis, validation, result, interpretation, and next step
- Shared reusable code goes in `src/`
- Reusable tests go in `tests/`

## Current State
- `README.md` contains the competition overview
- `docs/ENVIRONMENT.md` contains local setup steps
- `.gitignore` blocks competition data from Git tracking
- `AGENTS.md` defines repository workflow
- `experiments/exp000_eda/` exists for initial data exploration
- `experiments/exp001_baseline/` exists with `README.md`, `train.py`, `submission_baseline.ipynb`, and `notes.md`

## Immediate Next Steps
- run the first Kaggle submission from `experiments/exp001_baseline/submission_baseline.ipynb`
- record the first public LB result in `experiments/exp001_baseline/README.md`
- compare `strict_plus_ratingless` against a wider training variant only after the first score
- move reusable data loading and feature code into `src/` once it appears twice
- add smoke tests in `tests/` for any shared code

## Resume Checklist
- read `README.md`
- read `AGENTS.md`
- read this file
- open the latest experiment folder in `experiments/`
- run `git status --short` before making changes

## Notes for Future Sessions
- keep commits small and scoped to one experiment or one shared-code change
- record local CV, public LB, and key observations in the experiment `README.md`
- if a promising approach appears, create a new experiment folder instead of overwriting the old one
