# exp001_baseline

## Objective
Establish a simple baseline for BirdCLEF 2026 that can be reproduced and improved in later experiments.

## Hypothesis
- A simple clip-level classifier trained on a conservative high-confidence subset can produce a usable first submission signal.
- Classes with missing rating metadata should be rescued rather than dropped entirely.

## Validation Setup
- Dataset split: stratified train/validation split inside `train.py`
- Metric: accuracy, log loss, and macro one-vs-rest ROC-AUC on the local validation split
- Training data used: `experiments/exp000_eda/high_confidence_candidates_rescued.csv`
- Input features: log-mel mean/std summary features from the first 5 seconds of each clip
- Model: multinomial logistic regression with standardized inputs
- Augmentation:
- Seed: `42`
- Runtime constraints: keep the first pass light enough to iterate quickly before building a Kaggle submission notebook

## Result
- Public LB: `0.623`
- Local CV: smoke-run only for pipeline verification; not used as the main selection signal for this experiment
- Submission file: `submission.csv` from Kaggle notebook `BirdCLEF 2026 Exp001 Baseline V6`
- Run date: 2026-03-15

## Interpretation
The end-to-end submission path works and produces a non-trivial public score. The conservative `strict_plus_ratingless` filtering strategy appears good enough for a first pass, but the current feature/model stack is too weak to be competitive. Most remaining headroom is likely in better audio representations and stronger sequence-aware models rather than more filtering logic.

## Next Step
- Keep: `strict_plus_ratingless` as the current default training subset
- Change: replace log-mel summary statistics with a model that preserves time structure
- Drop: spending more time on placeholder submission plumbing for this experiment

## Notes
Add links to related scripts, configs, notebooks, or commits for this experiment.

- Training script: `train.py`
- Scratch notes: `notes.md`
- Submission notebook: `submission_baseline.ipynb`
- Expected artifacts: `artifacts/baseline_model.joblib`, `artifacts/metrics.json`
