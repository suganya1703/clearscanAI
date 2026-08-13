# ClearScan AI

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**
Team: Silicon Soup — Suganya S, Deepika S M, Brindha P

## Problem

Wafer / die inspection images are frequently degraded by noise, motion blur, defocus blur,
and low resolution — reducing the accuracy of downstream automated defect detection.

## Solution

ClearScan AI is a U-Net based image restoration pipeline that takes a degraded inspection
image and reconstructs a high-fidelity version, improving downstream defect-detection accuracy.

```
Degraded Image → U-Net Encoder-Decoder (skip connections) → Restored Image
```

## Project Structure

```
clearscan-ai/
├── data/
│   └── prepare_dataset.py     # generates synthetic degraded/clean pairs
├── src/
│   ├── model.py                # U-Net architecture
│   ├── dataset.py              # PyTorch Dataset / DataLoader
│   ├── train.py                # training loop
│   ├── evaluate.py             # PSNR / SSIM evaluation
│   └── infer.py                # run restoration on a single image
├── demo/
│   └── app.py                  # Streamlit demo: upload degraded image → see restored output
├── notebooks/
│   └── quickstart.ipynb        # end-to-end walkthrough
├── assets/                     # sample images (add your own wafer images here)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic degraded/clean training pairs from sample images in assets/
python data/prepare_dataset.py --input_dir assets/clean --output_dir data/pairs

# 2. Train the restoration model
python src/train.py --data_dir data/pairs --epochs 20 --out checkpoints/clearscan.pt

# 3. Evaluate
python src/evaluate.py --checkpoint checkpoints/clearscan.pt --data_dir data/pairs

# 4. Run the interactive demo
streamlit run demo/app.py
```

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) — pixel-level fidelity vs. ground truth
- **SSIM** (Structural Similarity Index) — structural fidelity, important for defect edges
- Downstream check: does restoration improve defect-detection accuracy on a sample detector

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch — U-Net based CNN |
| Data processing | OpenCV, NumPy, Albumentations |
| Evaluation | scikit-image (PSNR, SSIM) |
| Demo | Streamlit |
| Deployment | ONNX / TorchScript export |

## License

MIT — for hackathon/educational use.

## Declaration

This idea is original and not plagiarized from any other source.
