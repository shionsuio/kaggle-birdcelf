# Environment Setup

## Local Setup
Use a local virtual environment. One simple setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name birdclef2026 --display-name "Python (birdclef2026)"
```

Then select `Python (birdclef2026)` in Jupyter Notebook or VS Code.

## Current Baseline Packages
- `pandas`, `numpy`: table and numeric work
- `matplotlib`, `seaborn`: EDA plots
- `jupyter`, `ipykernel`: notebook workflow
- `scikit-learn`: basic validation helpers
- `librosa`, `soundfile`: audio loading and feature extraction
- `tqdm`: progress bars

## Notes
- Keep heavyweight training dependencies out of the base environment until they are actually needed.
- If a package is only used for one experiment, document that inside the experiment folder.
- Do not install packages globally for this project.
