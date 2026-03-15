"""Training entry point for exp003_cnn."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
DEFAULT_METADATA = ROOT / "experiments" / "exp000_eda" / "high_confidence_candidates_rescued.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact CNN BirdCLEF baseline.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-class row limit for smoke runs.")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--time-frames", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore cached spectrogram tensors and rebuild them.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    return np.vstack([np.interp(x_new, x_old, row) for row in spec])


def extract_logmel(path: Path, sample_rate: int, duration: float, n_mels: int, time_frames: int) -> np.ndarray:
    y, sr = load_audio(path, sample_rate=sample_rate, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    resized = resize_time_axis(log_mel, target_frames=time_frames)
    norm = (resized - resized.mean()) / (resized.std() + 1e-6)
    return norm.astype(np.float32)


def cache_dir(sample_rate: int, duration: float, n_mels: int, time_frames: int) -> Path:
    duration_token = str(duration).replace(".", "p")
    return ARTIFACT_DIR / "cache" / f"sr{sample_rate}_dur{duration_token}_mel{n_mels}_frames{time_frames}"


def cached_logmel(
    filename: str,
    sample_rate: int,
    duration: float,
    n_mels: int,
    time_frames: int,
    rebuild_cache: bool,
) -> np.ndarray:
    cache_root = cache_dir(sample_rate, duration, n_mels, time_frames)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / (Path(filename).stem + ".npy")
    if cache_path.exists() and not rebuild_cache:
        return np.load(cache_path)

    audio_path = ROOT / "train_audio" / filename
    spec = extract_logmel(audio_path, sample_rate, duration, n_mels, time_frames)
    np.save(cache_path, spec)
    return spec


def build_tensor(
    df: pd.DataFrame,
    sample_rate: int,
    duration: float,
    n_mels: int,
    time_frames: int,
    rebuild_cache: bool,
) -> np.ndarray:
    specs: list[np.ndarray] = []
    for filename in tqdm(df["filename"], desc="Loading log-mel tensors"):
        specs.append(
            cached_logmel(
                filename,
                sample_rate=sample_rate,
                duration=duration,
                n_mels=n_mels,
                time_frames=time_frames,
                rebuild_cache=rebuild_cache,
            )
        )
    array = np.stack(specs)
    return array[:, None, :, :]


class SpectrogramDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    classes = np.unique(y_true)
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch_y.numpy())
    return np.concatenate(targets), np.concatenate(probs)


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
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

    x = build_tensor(
        df,
        args.sample_rate,
        args.duration,
        args.n_mels,
        args.time_frames,
        args.rebuild_cache,
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

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=effective_test_size,
        random_state=args.random_state,
        stratify=y,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(SpectrogramDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(SpectrogramDataset(x_valid, y_valid), batch_size=args.batch_size, shuffle=False)

    model = SmallCNN(num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss: {epoch_loss:.4f}")

    y_true, valid_proba = evaluate(model, valid_loader, device)
    valid_pred = np.argmax(valid_proba, axis=1)
    metrics = {
        "metadata_path": str(metadata_path),
        "rows": int(len(df)),
        "classes": int(n_classes),
        "sample_rate": int(args.sample_rate),
        "duration": float(args.duration),
        "n_mels": int(args.n_mels),
        "time_frames": int(args.time_frames),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "accuracy": float(accuracy_score(y_true, valid_pred)),
        "macro_ovr_auc": macro_ovr_auc(y_true, valid_proba),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_classes": label_encoder.classes_.tolist(),
            "config": {
                "sample_rate": args.sample_rate,
                "duration": args.duration,
                "n_mels": args.n_mels,
                "time_frames": args.time_frames,
            },
        },
        ARTIFACT_DIR / "cnn_model.pt",
    )
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {ARTIFACT_DIR / 'cnn_model.pt'}")
    print(f"Saved metrics to {ARTIFACT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
