"""
Run ClearScan AI restoration on a single image.

Usage:
    python infer.py --checkpoint checkpoints/clearscan.pt --image path/to/degraded.png --out restored.png
"""
import argparse

import cv2
import numpy as np
import torch

from model import UNetRestorer


def restore_image(model, img_bgr, device, img_size=256):
    orig_h, orig_w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)[0].permute(1, 2, 0).cpu().numpy()

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    out = cv2.resize(out, (orig_w, orig_h))
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="restored.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetRestorer().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    img = cv2.imread(args.image)
    restored = restore_image(model, img, device)
    cv2.imwrite(args.out, restored)
    print(f"Saved restored image to {args.out}")


if __name__ == "__main__":
    main()
