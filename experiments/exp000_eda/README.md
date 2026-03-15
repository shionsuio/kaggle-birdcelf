# exp000_eda

## Objective
Inspect the BirdCLEF 2026 data layout, label structure, and obvious data quality risks before building the first baseline.

## Questions
- How many target classes are in submission?
- How imbalanced is `primary_label`?
- How often do `secondary_labels` appear?
- What metadata fields are most useful for filtering?
- How dense are labels in `train_soundscapes_labels.csv`?
- Are there obvious mismatches between training labels and submission targets?

## Outputs
- class distribution summary
- metadata summary
- label coverage checks
- notes on risks and next actions for baseline modeling

## Files
- `eda.py`: script entry point for quick tabular EDA
- `notes.md`: running observations

## Next Step
Use this experiment to decide the first reproducible modeling baseline in `exp001_baseline`.
