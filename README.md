# DeepVision Crowd Monitor

**Status: research prototype / implementation scaffold — not a validated real-time surveillance system.**

DeepVision explores crowd density estimation and threshold-based overcrowding alerts. This repository reconstructs an **image inference demo** from the three-page project specification; it is **not a recovery of original trained code or weights**.

## What is implemented

- CSRNet-style PyTorch architecture with VGG16 frontend and dilated convolution backend
- Load a *compatible, trusted* trained checkpoint (`.pth`/`.pt`)
- Estimate count as sum of the predicted density map (only meaningful with correctly trained weights)
- Visualize density heatmap and configurable count threshold alert
- Streamlit image-upload interface

## Planned / not yet implemented

- Training pipeline and ShanghaiTech dataset loaders
- Validated MAE/RMSE evaluation and measured accuracy
- Live CCTV/video frame processing and real-time performance benchmarks
- Zone-level density alerts, SMTP/Twilio, Docker, CUDA benchmarking

These are proposed milestones in the project PDF, not features to claim as finished.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You need a trained CSRNet checkpoint whose architecture and target density normalization match this implementation. Without one, the app shows an explicit warning and does not generate fake crowd counts. Checkpoints should come from trusted sources.

## Project pipeline

`Camera/video (planned) → frame extraction (planned) → preprocessing → CSRNet density map → sum density → count threshold → heatmap / alert`

## Dataset

ShanghaiTech Crowd Counting Dataset is mentioned in the source project specification. The dataset is **not included** here. Check dataset licensing and usage conditions before obtaining it.

## Limitations and safety

This is not appropriate for operational public-safety decisions. Camera angle, scale, domain shift, occlusion, and model calibration can substantially affect estimates. Density sum is not a calibrated people count unless the checkpoint was trained with count-preserving density targets.

## Origin

Reconstructed from *DeepVision Crowd Monitor: AI for Density Estimation and Overcrowding Detection*, a three-page project plan supplied by the project author. No training results or deployed-system claims are asserted.

## Author

Dev Rai — B.Tech, Artificial Intelligence & Machine Learning
