#!/usr/bin/env python3
"""
prepare_eatingsound_for_audioldm_local.py

目的:
 - 既にダウンロード済みの Eating Sound Collection を読み込み
 - 音声を指定サンプルレートにリサンプリングし mono に統一、正規化
 - 学習用マニフェスト (CSV) を作成: columns = [wav_path, caption, duration]

使い方 (例):
    python preprocessor/execute.py \
      --raw-dir ./audio_dataset \
      --out-dir ./data/dataset/kaggle_dataset/processed \
      --sample-rate 16000 \
      --manifest ./data/dataset/kaggle_dataset/manifest.csv
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import soundfile as sf
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_eatingsound")


# -----------------------
# Audio utils
# -----------------------
@dataclass
class AudioMeta:
    src_path: Path
    out_path: Path
    duration: float
    label: Optional[str]
    caption: str


def discover_audio_files(
    root: Path, exts: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def normalize_audio(wave: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(wave))
    if peak < 1e-8:
        return wave
    return wave / peak


def preprocess_and_save(src: Path, dst: Path, sample_rate: int) -> Tuple[float, int]:
    waveform, sr = torchaudio.load(src)
    waveform = waveform.numpy()
    # to mono
    if waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0, keepdims=True)
    # resample
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(torch_from_numpy(waveform)).numpy()
    # normalize
    waveform = normalize_audio(waveform)
    # save
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), waveform.squeeze(), sample_rate, subtype="PCM_16")
    duration = waveform.shape[-1] / float(sample_rate)
    return duration, sample_rate


def torch_from_numpy(arr: np.ndarray):
    import torch

    return torch.from_numpy(arr)


# -----------------------
# Caption logic
# -----------------------
def infer_label_from_path(p: Path) -> Optional[str]:
    parts = [part.lower() for part in p.parts]
    if len(parts) >= 2:
        return parts[-2]
    stem = p.stem.lower()
    if "_" in stem:
        return stem.split("_")[0]
    return None


def make_caption_for_label(label: Optional[str]) -> str:
    if not label:
        return "sound of someone eating food"
    return f"sound of someone eating {label.replace('_', ' ')}"


# -----------------------
# Manifest
# -----------------------
def build_manifest(rows: List[AudioMeta], out_manifest: Path):
    df = pd.DataFrame(
        [
            {"wav_path": str(r.out_path), "caption": r.caption, "duration": r.duration}
            for r in rows
        ]
    )
    df.to_csv(out_manifest, index=False)
    logger.info("Manifest saved to %s (%d rows)", out_manifest, len(df))


# -----------------------
# Main pipeline
# -----------------------
def prepare(
    raw_dir: Path,
    processed_dir: Path,
    sample_rate: int,
    manifest_path: Path,
):
    audio_files = list(discover_audio_files(raw_dir))
    if not audio_files:
        raise RuntimeError(f"No audio files found under {raw_dir}")

    logger.info("Found %d audio files for processing", len(audio_files))

    rows: List[AudioMeta] = []
    for src in tqdm(audio_files, desc="Processing audio"):
        try:
            rel = src.relative_to(raw_dir)
        except Exception:
            rel = src.name
        out_path = processed_dir / rel.with_suffix(".wav")
        duration, sr = preprocess_and_save(src, out_path, sample_rate)
        label = infer_label_from_path(src)
        caption = make_caption_for_label(label)
        rows.append(AudioMeta(src, out_path, duration, label, caption))

    build_manifest(rows, manifest_path)
    logger.info("Preparation complete. Processed data in: %s", processed_dir)


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare Eating Sound dataset (already downloaded) for AudioLDM2 fine-tuning"
    )
    p.add_argument(
        "--raw-dir",
        required=True,
        help="Path to the downloaded Eating Sound dataset root",
    )
    p.add_argument(
        "--out-dir",
        default="./eatingsound_processed",
        help="Where to save processed wav files",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for AudioLDM2",
    )
    p.add_argument(
        "--manifest", default="manifest.csv", help="Output manifest CSV path"
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        prepare(
            raw_dir=Path(args.raw_dir),
            processed_dir=Path(args.out_dir),
            sample_rate=args.sample_rate,
            manifest_path=Path(args.manifest),
        )
    except Exception as e:
        logger.exception("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
