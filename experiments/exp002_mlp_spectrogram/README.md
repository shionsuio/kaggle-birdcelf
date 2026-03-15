# exp002_mlp_spectrogram

## Objective
Improve on `exp001_baseline` by replacing clip-level summary statistics with a model that preserves coarse time structure from the spectrogram.

## Hypothesis
- Flattened log-mel spectrogram patches should retain more temporal information than mean/std summary features.
- A lightweight MLP can improve the first baseline without adding deep learning framework complexity.
- The current `strict_plus_ratingless` subset is still a reasonable default for the next iteration.

## Validation Setup
- Dataset split: stratified train/validation split inside `train.py`
- Metric: accuracy, log loss, and macro one-vs-rest ROC-AUC on the local validation split
- Training data used: `experiments/exp000_eda/high_confidence_candidates_rescued.csv`
- Input features: fixed-size log-mel spectrogram patch from the first 5 seconds of each clip
- Model: `sklearn.neural_network.MLPClassifier`
- Augmentation:
- Seed: `42`
- Runtime constraints: stay lightweight enough for local iteration before deciding whether to move to a CNN or torch-based model

## Result
- Public LB:
- Local CV:
- Submission file:
- Run date:

## Interpretation
Compare directly against `exp001_baseline`. The key question is whether preserving time structure helps enough to justify the larger feature space.

## Next Step
- Keep:
- Change:
- Drop:

## Notes
- Training script: `train.py`
- Scratch notes: `notes.md`
