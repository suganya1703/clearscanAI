"""
Evaluate ClearScan AI restoration quality using PSNR and SSIM.

Usage:
    python evaluate.py --checkpoint checkpoints/clearscan.pt --data_dir data/pairs
"""
import argparse

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset import RestorationDataset
from model import UNetRestorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetRestorer().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = RestorationDataset(args.data_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    psnr_scores, ssim_scores = [], []
    baseline_psnr_scores = []

    with torch.no_grad():
        for degraded, clean in loader:
            degraded, clean = degraded.to(device), clean.to(device)
            restored = model(degraded)

            r = restored[0].permute(1, 2, 0).cpu().numpy()
            c = clean[0].permute(1, 2, 0).cpu().numpy()
            d = degraded[0].permute(1, 2, 0).cpu().numpy()

            psnr_scores.append(psnr(c, r, data_range=1.0))
            ssim_scores.append(ssim(c, r, data_range=1.0, channel_axis=2))
            baseline_psnr_scores.append(psnr(c, d, data_range=1.0))

    print(f"Restored  -> PSNR: {np.mean(psnr_scores):.2f} dB | SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Degraded  -> PSNR: {np.mean(baseline_psnr_scores):.2f} dB (baseline, before restoration)")
    print(f"PSNR improvement: {np.mean(psnr_scores) - np.mean(baseline_psnr_scores):.2f} dB")


if __name__ == "__main__":
    main()
