"""Training entry point for exp002_mlp_spectrogram."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
DEFAULT_METADATA = ROOT / "experiments" / "exp000_eda" / "high_confidence_candidates_rescued.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spectrogram MLP BirdCLEF baseline.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-class row limit for smoke runs.")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--time-frames", type=int, default=128, help="Resize spectrogram to this many frames.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-class-count", type=int, default=2)
    return parser.parse_args()


def resolve_metadata_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = ROOT / "train.csv"
    print(f"Metadata file {path} not found. Falling back to {fallback}.")
    return fallback


def load_audio(path: Path, sample_rate: int, duration: float) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    frames = int(duration * info.samplerate)
    y, sr = sf.read(path, start=0, frames=frames, dtype="float32")
    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    target_length = int(sample_rate * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y, sr


def resize_time_axis(spec: np.ndarray, target_frames: int) -> np.ndarray:
    current_frames = spec.shape[1]
    if current_frames == target_frames:
        return spec
    x_old = np.linspace(0, 1, current_frames)
    x_new = np.linspace(0, 1, target_frames)
    resized = np.vstack([np.interp(x_new, x_old, row) for row in spec])
    return resized


def extract_features(path: Path, sample_rate: int, duration: float, n_mels: int, time_frames: int) -> np.ndarray:
    y, sr = load_audio(path, sample_rate=sample_rate, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    resized = resize_time_axis(log_mel, target_frames=time_frames)
    return resized.astype(np.float32).reshape(-1)


def build_feature_matrix(
    df: pd.DataFrame,
    sample_rate: int,
    duration: float,
    n_mels: int,
    time_frames: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for filename in tqdm(df["filename"], desc="Extracting spectrogram features"):
        audio_path = ROOT / "train_audio" / filename
        features.append(
            extract_features(
                audio_path,
                sample_rate=sample_rate,
                duration=duration,
                n_mels=n_mels,
                time_frames=time_frames,
            )
        )
    return np.vstack(features)


def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return None
    aucs: list[float] = []
    for class_id in classes:
        binary_true = (y_true == class_id).astype(int)
        positives = int(binary_true.sum())
        negatives = int((1 - binary_true).sum())
        if positives == 0 or negatives == 0:
            continue
        try:
            aucs.append(roc_auc_score(binary_true, proba[:, class_id]))
        except ValueError:
            continue
    if not aucs:
        return None
    return float(np.mean(aucs))


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = resolve_metadata_path(args.metadata)
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows from {metadata_path}")

    if args.limit is not None:
        df = df.groupby("primary_label", group_keys=False).head(max(1, args.limit)).copy()
        print(f"Applied per-class limit: {args.limit}. Rows now: {len(df)}")

    class_counts = df["primary_label"].value_counts()
    keep_labels = class_counts[class_counts >= args.min_class_count].index
    dropped_classes = len(class_counts) - len(keep_labels)
    if dropped_classes:
        df = df.loc[df["primary_label"].isin(keep_labels)].copy()
        print(f"Dropped {dropped_classes} classes with fewer than {args.min_class_count} rows.")

    X = build_feature_matrix(
        df,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
        time_frames=args.time_frames,
    )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["primary_label"])

    n_samples = len(df)
    n_classes = len(label_encoder.classes_)
    effective_test_size = args.test_size
    min_test_fraction = n_classes / n_samples
    if effective_test_size < min_test_fraction:
        effective_test_size = min(0.5, math.ceil(min_test_fraction * 1000) / 1000)
        print(
            "Adjusted test_size to "
            f"{effective_test_size:.3f} so validation has at least one sample per class."
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=effective_test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(512, 256),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    batch_size=128,
                    max_iter=50,
                    early_stopping=True,
                    random_state=args.random_state,
                    verbose=True,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    valid_proba = model.predict_proba(X_valid)
    valid_pred = np.argmax(valid_proba, axis=1)
    metrics = {
        "metadata_path": str(metadata_path),
        "rows": int(len(df)),
        "classes": int(len(label_encoder.classes_)),
        "sample_rate": int(args.sample_rate),
        "duration": float(args.duration),
        "n_mels": int(args.n_mels),
        "time_frames": int(args.time_frames),
        "accuracy": float(accuracy_score(y_valid, valid_pred)),
        "log_loss": float(log_loss(y_valid, valid_proba)),
        "macro_ovr_auc": macro_ovr_auc(y_valid, valid_proba),
    }

    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "config": {
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "n_mels": args.n_mels,
            "time_frames": args.time_frames,
        },
    }
    joblib.dump(payload, ARTIFACT_DIR / "mlp_spectrogram_model.joblib")
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {ARTIFACT_DIR / 'mlp_spectrogram_model.joblib'}")
    print(f"Saved metrics to {ARTIFACT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
