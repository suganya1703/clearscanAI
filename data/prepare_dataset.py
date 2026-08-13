"""
Generate synthetic degraded/clean image pairs for training ClearScan AI.

Given a folder of clean semiconductor inspection images, this script creates
degraded versions using:
  - Gaussian noise (sensor noise)
  - Gaussian / motion blur (defocus / scan-speed blur)
  - Downsampling + upsampling (resolution loss)

Usage:
    python prepare_dataset.py --input_dir assets/clean --output_dir data/pairs
"""
import argparse
import os
import random

import cv2
import numpy as np


def add_gaussian_noise(img, sigma_range=(5, 25)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(img, kernel_range=(3, 9)):
    k = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))  # odd kernel
    if random.random() < 0.5:
        return cv2.GaussianBlur(img, (k, k), 0)
    else:
        # simple motion blur
        kernel = np.zeros((k, k))
        kernel[(k - 1) // 2, :] = np.ones(k)
        kernel /= k
        return cv2.filter2D(img, -1, kernel)


def downsample_upsample(img, scale_range=(0.25, 0.6)):
    scale = random.uniform(*scale_range)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade(img):
    out = img.copy()
    if random.random() < 0.85:
        out = add_blur(out)
    if random.random() < 0.85:
        out = add_gaussian_noise(out)
    if random.random() < 0.6:
        out = downsample_upsample(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder of clean images")
    parser.add_argument("--output_dir", required=True, help="Where to write clean/ and degraded/ subfolders")
    parser.add_argument("--variants_per_image", type=int, default=3)
    args = parser.parse_args()

    clean_out = os.path.join(args.output_dir, "clean")
    degraded_out = os.path.join(args.output_dir, "degraded")
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(degraded_out, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]
    if not files:
        print(f"No images found in {args.input_dir}. Add sample wafer/chip images there first.")
        return

    count = 0
    for fname in files:
        img_path = os.path.join(args.input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        base = os.path.splitext(fname)[0]
        for v in range(args.variants_per_image):
            degraded = degrade(img)
            cv2.imwrite(os.path.join(clean_out, f"{base}_{v}.png"), img)
            cv2.imwrite(os.path.join(degraded_out, f"{base}_{v}.png"), degraded)
            count += 1

    print(f"Generated {count} degraded/clean pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
