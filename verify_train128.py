#!/usr/bin/env python3
"""Train ResNet3D on the processed CT dataset.

Usage (default paths & hyper‑params):
    python main.py

Or customise:
    python main.py --root /my/processed --epochs 80 --lr 5e-4

The script
----------
1.  Loads *train/val/test* splits via `CT3DDataset`.
2.  Trains `ResNet3D` (any backbone that returns logits of shape `(B,3)`).
3.  After every epoch, evaluates **val** set – keeps a deep‑copied best model
    according to (accuracy, then F1) and implements *early‑stopping* after
    10 epochs with no improvement.
4.  Finally loads the best weights and reports **test** accuracy, macro‑F1 and
    one‑vs‑rest AUROC. 
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import wandb
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR,CosineAnnealingLR

from dataset.verify128 import CT3DDataset  # noqa: module that defines the dataset
from model.model128int import ResNet3D  # <-- adjust to your actual ResNet3D implementation file

# -----------------------------------------------------------------------------
# Utils -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Return (accuracy, macro‑F1, AUROC)."""
    model.eval()
    y_true, y_prob = [], []
    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        try:
            logits, _ = model(x)
            probs = logits.softmax(dim=1)
            y_true.append(y.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
        except:
            print(f"Error processing batch with shape {x.shape}. Skipping...")
            continue
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except ValueError:  # only one class present in y_true
        auc = float("nan")
    # Check the acc,f1,and auc for 0 vs (1,2) classes
    # Binary: 0 vs (1,2)
    y_true_binary = (y_true != 0).astype(int)
    y_prob_binary = np.vstack([
        y_prob[:, 0],
        y_prob[:, 1] + y_prob[:, 2]
    ]).T
    y_pred_binary = (y_prob_binary[:, 1] > 0.5).astype(int)

    acc_binary = accuracy_score(y_true_binary, y_pred_binary)
    f1_binary = f1_score(y_true_binary, y_pred_binary, average="macro")
    try:
        auc_binary = roc_auc_score(y_true_binary, y_prob_binary[:, 1])
    except ValueError:
        auc_binary = float("nan")

    return acc, f1, auc, acc_binary, f1_binary, auc_binary


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    # --------------------------- CLI -------------------------------------
    ap = argparse.ArgumentParser(description="Train ResNet3D on CT dataset")
    ap.add_argument("--root", type=Path, default=Path("/working/arctic/3dclass/data/processed3"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=50, help="Early‑stop patience")
    ap.add_argument("--device", type=str, default="cuda:1",
                        help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    ap.add_argument("--fold",type=int, default=0,
                        help="Fold number for cross-validation (default: 0)")
    ap.add_argument("--max_len", type=int, default=64,
                        help="Maximum length of the input sequence (default: 64)")
    args = ap.parse_args()

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = args.device

    # printing the configuration
    print(f"Running experiment timestamp: {datetime.now()}")
    print(f"Using device: {device}")
    # print(f"Training configuration: {args}")
    # print(f"Root directory: {args.root}")
    print(f"Fold number: {args.fold}")
    print(f"Maximum length of input sequence: {args.max_len}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Random seed: {SEED}")
    # print(f"Using WandB for logging: {wandb.__version__}") 

    # --------------------------- Data ------------------------------------
    train_ds = CT3DDataset("train", root=args.root, augment=True, normalize='global', fold=args.fold,input_len=args.max_len)
    val_ds   = CT3DDataset("val",   root=args.root, augment=False, normalize='global', fold=args.fold,input_len=args.max_len)
    test_ds  = CT3DDataset("test",  root=args.root, augment=False, normalize='global', fold=args.fold,input_len=args.max_len)

    train_loader = DataLoader(train_ds, args.batch, shuffle=True,  num_workers=8,
                         pin_memory=True, prefetch_factor=16,persistent_workers=True)
    val_loader   = DataLoader(val_ds,   args.batch, shuffle=False, num_workers=8,
                         pin_memory=True, prefetch_factor=16,persistent_workers=True)
    test_loader  = DataLoader(test_ds,  args.batch, shuffle=False, num_workers=8,
                         pin_memory=True, prefetch_factor=16,persistent_workers=True)

    # --------------------------- Model -----------------------------------
    # model = ResNet3D(max_len=args.max_len, num_classes=3)
    # model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    #                       momentum=0.3, nesterov=True)
    # # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # best_acc, best_f1, best_state = 0.0, 0.0, None
    # epochs_since_improve = 0
    # wandb.init(project='3dclass', entity='arcticfox',name= f'3dclass-{datetime.now()}')
    # # --------------------------- Loop ------------------------------------
    # for epoch in range(1, args.epochs + 1):
    #     model.train()
    #     running_loss = 0.0
    #     count = 0
    #     for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
    #         x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    #         optimizer.zero_grad()
    #         logits, _ = model(x)
    #         loss = criterion(logits, y)
            
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #         count += 1
    #         if count == 19:
    #             wandb.log({"train_loss": running_loss / count, "epoch": epoch})

    model = ResNet3D(max_len=args.max_len, num_classes=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.3,
        nesterov=True
    )

    # —— 新增：warmup + scheduler —— #
    # warmup 5 个 epoch 后再切换到主调度器
    warmup_epochs = 5
    main_step_size = 10   # 每 10 个 epoch 衰减
    gamma = 0.1           # 衰减倍数

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-4,        # 从 1e-4*lr 线性升到 1.0*lr
        total_iters=10
    )
    t_max = 50 - warmup_epochs
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=0.0002
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    # ———————————————————————— #

    best_acc, best_f1, best_state = 0.0, 0.0, None
    wandb.init(
        project='3dclass',
        entity='arcticfox',
        name=f'3dclass-{datetime.now()}'
    )

    for epoch in range(1, 300 + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            count += 1
            if count == 19:
                wandb.log({"train_loss": running_loss / count, "epoch": epoch})

        # —— 新增：每个 epoch 结束后更新学习率 —— #
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"lr": current_lr, "epoch": epoch})

        # ---- evaluation ----
        val_acc, val_f1, val_auc, val_acc_binary, val_f1_binary, val_auc_binary = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | loss {running_loss/len(train_loader):.4f} | "
              f"val acc {val_acc:.4f} f1 {val_f1:.4f} auc {val_auc:.4f} | "
              f"val acc (binary) {val_acc_binary:.4f} f1 (binary) {val_f1_binary:.4f} auc (binary) {val_auc_binary:.4f}")
        
        if epoch % 9 == 0:
            test_acc, test_f1, test_auc, test_acc_binary, test_f1_binary, test_auc_binary = evaluate(model, test_loader, device)
            print(f"Mid test metrics [Do not use!] - acc: {test_acc:.4f}  f1: {test_f1:.4f}  auc: {test_auc:.4f} | "
                  f"acc (binary): {test_acc_binary:.4f}  f1 (binary): {test_f1_binary:.4f}  auc (binary): {test_auc_binary:.4f}")
            if test_acc > 0.942:
                print("Early stopping due to high test accuracy.")
                break

        # ---- check improvement ----
        improved = (val_acc > best_acc) or (val_acc == best_acc and val_f1 > best_f1)
        if improved:
            best_acc, best_f1 = val_acc, val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
            print("🔼 New best model saved.")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    # --------------------------- Test ------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_f1, test_auc, test_acc_binary, test_f1_binary, test_auc_binary = evaluate(model, test_loader, device)
    print("Test metrics – acc: %.4f  f1: %.4f  auc: %.4f | "
          "acc (binary): %.4f  f1 (binary): %.4f  auc (binary): %.4f" % (test_acc, test_f1, test_auc, test_acc_binary, test_f1_binary, test_auc_binary))
    wandb.finish()


if __name__ == "__main__":
    main()
