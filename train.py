"""Joint training of scenario classifier and covariance matrix reconstructor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mimo_doa.dataset import SCENARIO_LABELS, SyntheticDOADataset
from mimo_doa.models import CovReconstructor, ScenarioClassifier
from mimo_doa.signal_model import ti_awr1843_geometry


def collate(batch):
    x_vec = torch.stack([b[0] for b in batch])
    cov_in = torch.stack([b[1] for b in batch])
    cov_clean = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    angles = torch.stack([b[4] for b in batch])
    snrs = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    return x_vec, cov_in, cov_clean, labels, angles, snrs


def train(
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    train_len: int,
    val_len: int,
    seed: int,
    device: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry = ti_awr1843_geometry()
    M = geometry.n_virtual

    train_ds = SyntheticDOADataset(geometry, length=train_len, seed=seed)
    val_ds = SyntheticDOADataset(geometry, length=val_len, seed=seed + 10_000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)

    classifier = ScenarioClassifier(n_virtual=M, n_classes=len(SCENARIO_LABELS)).to(device)
    reconstructor = CovReconstructor(n_virtual=M).to(device)

    opt = torch.optim.Adam(
        list(classifier.parameters()) + list(reconstructor.parameters()), lr=lr
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mse": []}

    for epoch in range(epochs):
        classifier.train(); reconstructor.train()
        running = 0.0
        for x_vec, cov_in, cov_clean, labels, _, _ in train_loader:
            x_vec = x_vec.to(device); cov_in = cov_in.to(device)
            cov_clean = cov_clean.to(device); labels = labels.to(device)

            logits = classifier(cov_in)
            cls_loss = ce(logits, labels)

            cov_pred = reconstructor(cov_in)
            non_multipath = labels != 2
            if non_multipath.any():
                rec_loss = mse(cov_pred[non_multipath], cov_clean[non_multipath])
            else:
                rec_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + 0.5 * rec_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * x_vec.size(0)
        sched.step()
        train_loss = running / max(len(train_ds), 1)

        classifier.eval(); reconstructor.eval()
        with torch.no_grad():
            val_running = 0.0
            correct = 0
            mse_running = 0.0
            n_non_mp = 0
            for x_vec, cov_in, cov_clean, labels, _, _ in val_loader:
                x_vec = x_vec.to(device); cov_in = cov_in.to(device)
                cov_clean = cov_clean.to(device); labels = labels.to(device)
                logits = classifier(cov_in)
                cls_loss = ce(logits, labels)
                cov_pred = reconstructor(cov_in)
                non_multipath = labels != 2
                if non_multipath.any():
                    rec_loss = mse(cov_pred[non_multipath], cov_clean[non_multipath])
                    mse_running += float(rec_loss.item()) * int(non_multipath.sum())
                    n_non_mp += int(non_multipath.sum())
                else:
                    rec_loss = torch.tensor(0.0, device=device)
                loss = cls_loss + 0.5 * rec_loss
                val_running += float(loss.item()) * x_vec.size(0)
                correct += int((logits.argmax(dim=1) == labels).sum())
        val_loss = val_running / max(len(val_ds), 1)
        val_acc = correct / max(len(val_ds), 1)
        val_mse = mse_running / max(n_non_mp, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_mse"].append(val_mse)
        print(f"epoch {epoch+1:02d}/{epochs}: train {train_loss:.4f} | "
              f"val {val_loss:.4f} | acc {val_acc:.3f} | recMSE {val_mse:.4f}")

    torch.save(classifier.state_dict(), out_dir / "classifier.pt")
    torch.save(reconstructor.state_dict(), out_dir / "reconstructor.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved weights to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-len", type=int, default=6000)
    p.add_argument("--val-len", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    train(args.out_dir, args.epochs, args.batch_size, args.lr,
          args.train_len, args.val_len, args.seed, args.device)


if __name__ == "__main__":
    main()
