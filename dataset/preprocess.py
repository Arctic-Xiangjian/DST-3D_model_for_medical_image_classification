#!/usr/bin/env python3
"""
CT study → 3‑D NumPy conversion + stratified split
=================================================

*   Converts each **leaf folder** of PNG slices into a single `(H,W,S)` volume;
*   Drops any study whose slices do **not all share the same shape** and prints
    its UID so you can inspect later;
*   Writes a 7 : 1 : 2 (train/val/test) split with preserved class ratios under
    `/v/ai/nobackup/arctic/3dclass/data/processed/…`.

```bash
python ct_preprocess_split.py                     # default locations
python ct_preprocess_split.py --root /raw --out /processed
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
from scipy.ndimage import zoom

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------------------------------------------------------------
LABEL_MAP = {"CP": 0, "NCP": 1, "Normal": 2}
SPLIT = (0.7, 0.1, 0.2)  # train, val, test
SEED = 42

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# def load_volume(study_dir: Path) -> np.ndarray:
#     """Stack all PNG slices into a 3‑D array; raise *ValueError* if any slice
#     has a differing 2‑D shape.
#     """
#     slice_paths = sorted(study_dir.glob("*.png"))
#     if not slice_paths:
#         raise RuntimeError(f"No PNG slices found in {study_dir}")
#     # resize to 512x512 if needed
#     # slices = [np.array(Image.open(p).convert("L")) for p in slice_paths]
#     slices = []
#     for p in slice_paths:
#         img = Image.open(p).convert("L")
#         size_img = img.size
#         if size_img[0] != size_img[1]:
#             raise ValueError("Inconsistent slice shapes within study")
#         if img.size != (512, 512):
#             img = img.resize((512, 512), Image.LANCZOS)
#         slices.append(np.array(img, dtype=np.float32))
#     first_shape = slices[0].shape
#     if any(s.shape != first_shape for s in slices[1:]):
#         raise ValueError("Inconsistent slice shapes within study")
#     if len(slices) < 64:
#         # resample to 64 slices by interpolation
        

#     return np.stack(slices, axis=-1)  # (H,W,S)


def load_volume(study_dir: Path) -> np.ndarray:
    """Load and stack PNG slices into a 3D volume with shape (H, W, 64).
    Resample if slice count < 64, raise ValueError if shape mismatch."""
    slice_paths = sorted(study_dir.glob("*.png"))
    if not slice_paths:
        raise RuntimeError(f"No PNG slices found in {study_dir}")

    slices = []
    for p in slice_paths:
        img = Image.open(p).convert("L")
        if img.size[0] != img.size[1]:
            raise ValueError("Non-square slice detected")
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.LANCZOS)
        slices.append(np.array(img, dtype=np.float32))

    first_shape = slices[0].shape
    if any(s.shape != first_shape for s in slices[1:]):
        raise ValueError("Inconsistent slice shapes within study")

    volume = np.stack(slices, axis=-1)  # (H, W, S)
    current_depth = volume.shape[-1]

    if current_depth < 64:
        # interpolate to 64 slices along depth axis
        zoom_factor = 64 / current_depth
        volume = zoom(volume, (1, 1, zoom_factor), order=1)  # linear interpolation
        print(f"⚠️  Resampled {study_dir.name} from {current_depth} to 64 slices.")
        if volume.shape[-1] != 64:
            raise ValueError(f"Failed to resample {study_dir.name} to 64 slices.")

    return volume  # shape guaranteed to be (512, 512, 64)

def gather_samples(root: Path) -> Tuple[List[Tuple[Path, int, str]], np.ndarray]:
    samples, labels = [], []
    for cls_name, cls_label in LABEL_MAP.items():
        class_dir = root / cls_name
        if not class_dir.is_dir():
            raise RuntimeError(f"Expected directory {class_dir} not found.")
        for patient_dir in class_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                uid = f"{study_dir.name}_{patient_dir.name}"
                samples.append((study_dir, cls_label, uid))
                labels.append(cls_label)
    return samples, np.asarray(labels, dtype=np.int32)


def stratified_indices(labels: np.ndarray):
    idx_all = np.arange(len(labels))
    train_idx, tmp_idx = train_test_split(idx_all, test_size=1 - SPLIT[0], stratify=labels, random_state=SEED)
    val_rel = SPLIT[1] / (1 - SPLIT[0])
    val_idx, test_idx = train_test_split(tmp_idx, test_size=1 - val_rel, stratify=labels[tmp_idx], random_state=SEED)
    return train_idx, val_idx, test_idx


def save_split(idx: np.ndarray, name: str, samples: List[Tuple[Path, int, str]], out_root: Path):
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for i in tqdm(idx, desc=f"Writing {name}"):
        study_dir, label, uid = samples[i]
        try:
            vol = load_volume(study_dir)
        except ValueError as e:
            print(f"⚠️  Skipping {uid} ({study_dir}): {e}")
            skipped += 1
            continue
        np.save(out_dir / f"{uid}.npy", {"data": vol, "label": label})

    if skipped:
        print(f"🔹 {skipped} studies dropped from {name} due to inconsistent slice shapes.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(root: Path, out_root: Path):
    print(f"Scanning dataset under {root} …")
    samples, labels = gather_samples(root)

    print(f"Found {len(samples)} CT studies in total.")
    train_idx, val_idx, test_idx = stratified_indices(labels)
    print(f"Split sizes – train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    save_split(train_idx, "train", samples, out_root)
    save_split(val_idx, "val", samples, out_root)
    save_split(test_idx, "test", samples, out_root)
    print("✅ Done – processed volumes saved to", out_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare 3‑D CT dataset")
    p.add_argument("--root", type=Path, default=Path("/v/ai/nobackup/arctic/3dclass/data/dataset_cleaned"))
    p.add_argument("--out", type=Path, default=Path("/working/arctic/3dclass/data/processed3"))
    args = p.parse_args()
    run(args.root, args.out)
