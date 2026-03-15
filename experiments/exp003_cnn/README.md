# exp003_cnn

## Objective
Improve on `exp001_baseline` by training a small CNN directly on log-mel spectrogram images instead of flattening or summarizing them.

## Hypothesis
- A compact 2D CNN can use local time-frequency structure more effectively than logistic regression or an MLP on flattened features.
- The current `strict_plus_ratingless` subset is still a reasonable starting point while model capacity increases.

## Validation Setup
- Dataset split: stratified train/validation split inside `train.py`
- Metric: accuracy and macro one-vs-rest ROC-AUC on the local validation split
- Training data used: `experiments/exp000_eda/high_confidence_candidates_rescued.csv`
- Input features: single-channel log-mel spectrogram from the first 5 seconds of each clip
- Model: small PyTorch CNN
- Augmentation:
- Seed: `42`
- Runtime constraints: keep the model compact enough for quick local iteration before adapting it to Kaggle inference

## Result
- Public LB:
- Local CV:
- Submission file:
- Run date:

## Interpretation
Compare directly against `exp001_baseline` and `exp002_mlp_spectrogram`. The main question is whether a small CNN already beats feature-engineered baselines.

## Next Step
- Keep:
- Change:
- Drop:

## Notes
- Training script: `train.py`
- Scratch notes: `notes.md`
