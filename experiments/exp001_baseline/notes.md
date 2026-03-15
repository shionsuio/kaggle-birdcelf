# exp001_baseline notes

## Quick Log
- Date: 2026-03-15
- Change: built the first full training and Kaggle submission path using `strict_plus_ratingless`
- Reason: validate that the repository structure, filtering logic, and Kaggle notebook path all work end to end
- Outcome: Kaggle public leaderboard score `0.623`, roughly around 330th place at the time of submission

## Observations
- The first meaningful bottleneck is model capacity, not repository plumbing.
- `strict_plus_ratingless` was a better default than strict-only or lightly relaxed filtering.
- Interactive Kaggle runs did not expose hidden test audio, but submit/rerun still produced a real scored submission.
- Placeholder outputs in interactive mode are not useful for quality assessment; the public leaderboard is the real signal.

## Follow-up
- Start `exp002` with a model that preserves time structure.
- Reuse the current Kaggle submission path and focus on improving feature/model quality.
- Compare against a wider training subset only after a stronger model baseline exists.
