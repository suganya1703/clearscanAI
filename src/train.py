"""
Train ClearScan AI restoration model.

Usage:
    python train.py --data_dir data/pairs --epochs 20 --out checkpoints/clearscan.pt
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RestorationDataset
from model import UNetRestorer


def ssim_l1_loss(pred, target, alpha=0.8):
    """Combined L1 + (1 - approx SSIM) loss. Simple SSIM approx via structural term."""
    l1 = torch.nn.functional.l1_loss(pred, target)
    # lightweight structural term: gradient similarity
    def grad(img):
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx, dy

    pdx, pdy = grad(pred)
    tdx, tdy = grad(target)
    grad_loss = torch.nn.functional.l1_loss(pdx, tdx) + torch.nn.functional.l1_loss(pdy, tdy)

    return alpha * l1 + (1 - alpha) * grad_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--out", default="checkpoints/clearscan.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RestorationDataset(args.data_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = UNetRestorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for degraded, clean in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            degraded, clean = degraded.to(device), clean.to(device)

            optimizer.zero_grad()
            restored = model(degraded)
            loss = ssim_l1_loss(restored, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
