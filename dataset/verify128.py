#!/usr/bin/env python3
"""
CT data pipeline
===============

* **Pre‑processing** – Converts raw PNG slices into 3‑D volumes and saves them
  to a processed directory. This is a one-time step.
* **CT3DDataset** – Loads the processed .npy volumes. It now contains an
  internal 5-fold cross-validation splitter that operates on a patient level,
  ensuring no patient's data appears in multiple sets within a fold.
  It dynamically creates train (70%), val (10%), and test (20%) splits
  for a given fold.

Update (2025-06-17)
-------------------
The CT3DDataset class has been refactored to handle patient-stratified
5-fold cross-validation internally, making the initial file-based split
obsolete for training/evaluation loops.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple, Literal, Dict

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchio as tio

# -----------------------------------------------------------------------------
# Pre‑processing helpers (for one-time PNG to .npy conversion)
# -----------------------------------------------------------------------------
LABEL_MAP = {"CP": 0, "NCP": 1, "Normal": 2}
SEED = 42
DEFAULT_OUT = Path("/v/ai/nobackup/arctic/3dclass/data/processed3") # You can change this

std_mean = {'fold_0': ([0.34351226687431335], [0.40813207626342773]), 'fold_1': ([0.34091901779174805], [0.40792718529701233]), 'fold_2': ([0.34541696310043335], [0.4086291491985321]), 'fold_3': ([0.34307584166526794], [0.4087154269218445]), 'fold_4': ([0.347238689661026], [0.4097311198711395])}



def _load_volume_raw(study_dir: Path) -> np.ndarray:
    paths = sorted(study_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG slices in {study_dir}")
    slices = [np.array(Image.open(p).convert("L")) for p in paths]
    if any(s.shape != slices[0].shape for s in slices[1:]):
        raise ValueError("Inconsistent slice shapes in study")
    return np.stack(slices, axis=-1)  # (H,W,D)


def _gather_samples(raw_root: Path) -> List[Tuple[Path, int, str]]:
    samples = []
    print("Gathering samples from raw data directory...")
    for cls_name, cls_label in LABEL_MAP.items():
        class_dir = raw_root / cls_name
        if not class_dir.is_dir(): continue
        for patient_dir in class_dir.iterdir():
            if not patient_dir.is_dir(): continue
            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir(): continue
                # UID combines study and patient ID for uniqueness
                uid = f"{study_dir.name}_{patient_dir.name}"
                samples.append((study_dir, cls_label, uid))
    return samples

def _save_all_as_npy(samples: List[Tuple[Path, int, str]], out_root: Path):
    """Saves all valid studies as .npy files into a single directory."""
    out_root.mkdir(parents=True, exist_ok=True)
    skipped = 0
    print(f"Converting {len(samples)} studies to .npy format...")
    for study_dir, lbl, uid in tqdm(samples, desc="Writing .npy files"):
        try:
            vol = _load_volume_raw(study_dir)
        except (ValueError, RuntimeError) as e:
            print(f"⚠️  Skipping {uid}: {e}")
            skipped += 1
            continue
        # Save as a dictionary for clarity
        np.save(out_root / f"{uid}.npy", {"data": vol, "label": lbl})
    if skipped:
        print(f"🔹  {skipped} studies were skipped due to errors.")


def prepare_dataset(raw_root: Path, out_root: Path = DEFAULT_OUT):
    """
    One-time data preparation step. Converts all PNG studies into .npy files
    and stores them in the output directory. The train/val/test split will
    be handled by the Dataset class on the fly.
    """
    samples = _gather_samples(raw_root)
    _save_all_as_npy(samples, out_root)

# -----------------------------------------------------------------------------
# PyTorch Dataset with Internal K-Fold Cross-Validation
# -----------------------------------------------------------------------------
NORM_MEAN, NORM_STD = 0.5, 0.5  # placeholders

class CT3DDataset(Dataset):
    """
    Dataset returning `(1,H,W,D)` tensors and integer labels.
    It performs a patient-stratified 5-fold split internally.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        *,
        root: Path = DEFAULT_OUT,
        fold: int = 0, # The fold index for cross-validation (0-4)
        augment: bool = False,
        normalize: str = "zscore",
        input_len: int = 16,
        mean_and_std: Dict[str, Tuple[List[float], List[float]]] = std_mean
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Processed data root not found: {self.root}")

        self.split = split
        self.augment = augment
        self.input_len = input_len

        if fold == 0:
            keys = 'fold_0'
            mean, std = mean_and_std[keys]
            self.mean = mean[0]
            self.std = std[0]
        elif fold == 1:
            keys = 'fold_1'
            mean, std = mean_and_std[keys]
            self.mean = mean[0]
            self.std = std[0]
        elif fold == 2:
            keys = 'fold_2'
            mean, std = mean_and_std[keys]
            self.mean = mean[0]
            self.std = std[0]
        elif fold == 3:
            keys = 'fold_3'
            mean, std = mean_and_std[keys]
            self.mean = mean[0]
            self.std = std[0]
        elif fold == 4:
            keys = 'fold_4'
            mean, std = mean_and_std[keys]
            self.mean = mean[0]
            self.std = std[0]
        else:
            raise ValueError(f"Invalid fold index: {fold}. Must be 0-4.")

        # --- Internal Patient-Level 5-Fold Splitting Logic ---
        
        # 1. Scan all processed .npy files and map them to their patient IDs.
        all_files = sorted(list(self.root.glob('**/*.npy')))
        if not all_files:
            raise RuntimeError(f"No .npy files found in {self.root}. Did you run the pre-processing step?")
            
        # The UID format is 'studyid_patientid'. We extract the patient ID.
        patient_map = {path: path.name.split('_')[1].split('.')[0] for path in all_files}
        
        # 2. Get a sorted list of unique patient IDs for reproducible splits.
        unique_patients = sorted(list(set(patient_map.values())))
        
        # 3. Create the 5-fold splits based on patient IDs.
        # A fixed random_state ensures the folds are the same every time.
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        
        patient_indices = list(kf.split(unique_patients))
        dev_idx, test_idx = patient_indices[fold]
        
        # Convert indices back to patient IDs
        dev_patients_np = np.array(unique_patients)[dev_idx]
        test_patients = set(np.array(unique_patients)[test_idx]) # This is our 20% test set
        
        # 4. Split the 80% development set into train (70%) and val (10%).
        # The relative split is 70:10, so val set is 1/8th (12.5%) of the dev set.
        train_patients_np, val_patients_np = train_test_split(
            dev_patients_np,
            test_size=0.125, # 10% / 80% = 0.125
            random_state=SEED,
            shuffle=True
        )
        train_patients = set(train_patients_np)
        val_patients = set(val_patients_np)
        
        # 5. Select the correct patient group based on the requested split mode.
        if self.split == "train":
            target_patients = train_patients
        elif self.split == "val":
            target_patients = val_patients
        else: # "test"
            target_patients = test_patients
            
        # 6. Filter the master file list to get the final paths for this dataset instance.
        self.paths = [path for path, patient_id in patient_map.items() if patient_id in target_patients]
        # --- End of Splitting Logic ---
        
        # --- TorchIO Transforms (same as your original code) ---
        base = [tio.RescaleIntensity((0, 1)),
                # resize to 128x128xinput_len
                tio.Resize((self.input_len,128, 128))]
        inten_aug = [
            tio.RandomNoise(0, (0, 0.05), p=0.2),
            tio.RandomBlur((0, 1), p=0.2),
            tio.RandomMotion(degrees=10, translation=10, num_transforms=2, image_interpolation='linear', p=0.2)
        ] if augment else []
        if normalize == "global":
            norm = [tio.Lambda(lambda x: (x - self.mean) / self.std)]
        elif normalize == "zscore":
            norm = [tio.ZNormalization()]
        else:
            norm = []
        spatial_aug = [
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        ] if augment else []
        self.transform = tio.Compose(base + inten_aug + norm + spatial_aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # This part is largely unchanged from your original implementation.
        item = np.load(self.paths[idx], allow_pickle=True).item()
        vol, label = item["data"], int(item["label"])

        # Depth crop BEFORE channel‑first conversion; depth axis == -1
        depth = vol.shape[-1]
        
        # For training, random crop. For val/test, center crop for consistency.
        if depth > self.input_len:
            if self.augment: # Random crop for training
                start = (depth - self.input_len) // 2
            else: # Center crop for validation/testing
                start = (depth - self.input_len) // 2
            vol = vol[:, :, start:start + self.input_len]

        # Convert to (1, D, H, W) for TorchIO
        tensor = torch.as_tensor(vol, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        subject = tio.Subject(image=tio.ScalarImage(tensor=tensor), label=label)
        subject = self.transform(subject)
        out_data = subject["image"].data
        
        # Transpose to your model's expected (1, H, W, D) format
        out_data = out_data.permute(0, 2, 3, 1)
        return out_data, torch.tensor(label, dtype=torch.long)

# -----------------------------------------------------------------------------
# CLI and Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare dataset and/or test loader")
    # Point this to your RAW dataset of PNGs
    ap.add_argument("--raw-root", type=Path, default=Path("./dataset_raw_example"))
    # Point this to where you want the processed .npy files to be saved
    ap.add_argument("--processed-root", type=Path, default=Path("/working/arctic/3dclass/data/processed2"))
    ap.add_argument("--run-prep", action="store_true", help="Run the one-time pre-processing step.")
    ap.add_argument("--verify-splits", action="store_true", help="Run a verification of the 5-fold split logic.")
    args = ap.parse_args()

    # --- Run one-time pre-processing if requested ---
    if args.run_prep:
        print("="*50)
        print("STEP 1: RUNNING ONE-TIME PRE-PROCESSING (PNG -> NPY)")
        print("="*50)
        prepare_dataset(args.raw_root, args.processed_root)

    # --- Verify the 5-fold logic if requested ---
    if args.verify_splits:
        print("\n" + "="*50)
        print("STEP 2: VERIFYING THE 5-FOLD PATIENT-LEVEL SPLIT")
        print("="*50)
        
        all_patient_ids = set()
        all_test_patients_in_folds = []
        
        # Gather all patients from the processed directory
        all_files_in_processed = sorted(list(args.processed_root.glob('**/*.npy')))
        if not all_files_in_processed:
            print("No .npy files found in processed_root. Please run with --run-prep first on a dummy dataset.")
        else:
            for f in all_files_in_processed:
                all_patient_ids.add(f.name.split('_')[1].split('.')[0])
            
            total_patient_count = len(all_patient_ids)
            print(f"Found {len(all_files_in_processed)} total files belonging to {total_patient_count} unique patients.")
            print("-" * 30)
            
            for fold_idx in range(5):
                print(f"--- FOLD {fold_idx} ---")
                train_ds = CT3DDataset("train", root=args.processed_root, fold=fold_idx)
                val_ds   = CT3DDataset("val", root=args.processed_root, fold=fold_idx)
                test_ds  = CT3DDataset("test", root=args.processed_root, fold=fold_idx)

                # print the number of files in each split
                print(f"Train files: {len(train_ds.paths)}")
                print(f"Val files:   {len(val_ds.paths)}")
                print(f"Test files:  {len(test_ds.paths)}")

                train_pids = {p.name.split('_')[1].split('.')[0] for p in train_ds.paths}
                val_pids   = {p.name.split('_')[1].split('.')[0] for p in val_ds.paths}
                test_pids  = {p.name.split('_')[1].split('.')[0] for p in test_ds.paths}
                
                print(f"Train patients: {len(train_pids)} (~{len(train_pids)/total_patient_count:.1%})")
                print(f"Val patients:   {len(val_pids)} (~{len(val_pids)/total_patient_count:.1%})")
                print(f"Test patients:  {len(test_pids)} (~{len(test_pids)/total_patient_count:.1%})")

                # Verify that patient sets are mutually exclusive
                assert train_pids.isdisjoint(val_pids), f"Fold {fold_idx}: Train/Val sets have overlapping patients!"
                assert train_pids.isdisjoint(test_pids), f"Fold {fold_idx}: Train/Test sets have overlapping patients!"
                assert val_pids.isdisjoint(test_pids), f"Fold {fold_idx}: Val/Test sets have overlapping patients!"
                print("Patient sets are mutually exclusive: OK.")
                all_test_patients_in_folds.extend(list(test_pids))

            print("-" * 30)
            # Verify that the 5 test folds cover all patients
            print(f"Total unique patients covered by 5 test folds: {len(set(all_test_patients_in_folds))}")
            assert total_patient_count == len(set(all_test_patients_in_folds)), "The test sets of the 5 folds do not cover all patients!"
            print("5-Fold test sets successfully cover all patients without overlap: OK.")