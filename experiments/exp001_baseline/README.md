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
- Public LB:
- Local CV:
- Submission file:
- Run date:

## Interpretation
Summarize what worked, what failed, and the most likely reason.

## Next Step
- Keep:
- Change:
- Drop:

## Notes
Add links to related scripts, configs, notebooks, or commits for this experiment.

- Training script: `train.py`
- Scratch notes: `notes.md`
- Submission notebook: `submission_baseline.ipynb`
- Expected artifacts: `artifacts/baseline_model.joblib`, `artifacts/metrics.json`
