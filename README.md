# DeepVision Crowd Monitor

### Crowd Density Estimation and Overcrowding Detection using Deep Learning

DeepVision is a computer vision project focused on estimating the number of people in crowded scenes using density maps. The idea is to explore how deep learning could help monitor busy public spaces and identify situations where crowd levels exceed a defined threshold.

I started working on this problem because traditional object detection can struggle in densely packed scenes, especially when people overlap or appear very small in the image.

## What the Project Does

The current implementation includes:

- A CSRNet-style neural network built with PyTorch
- Image preprocessing and density-map inference
- Estimated crowd counts from predicted density maps
- A configurable threshold for overcrowding alerts
- A simple Streamlit interface for uploading and analyzing images

**Note:** Crowd-count predictions require compatible trained model weights. This repository currently provides the model architecture and inference code, not a trained checkpoint.

## How It Works

1. An image is uploaded through the Streamlit interface.
2. The image is prepared for model inference.
3. A CSRNet-style network generates a density map.
4. The predicted density values are summed to estimate crowd count.
5. The estimated count is compared against a user-defined threshold.
6. The interface displays the density visualization and alert status.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| PyTorch | Neural network architecture and inference |
| CSRNet | Crowd density estimation |
| OpenCV | Computer vision utilities |
| NumPy | Numerical processing |
| Streamlit | User interface |

## Running the Project

Clone the repository:

```bash
git clone https://github.com/DevCodesAI/DeepVision-Crowd-Monitor.git
cd DeepVision-Crowd-Monitor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

A compatible trained CSRNet checkpoint is required to generate meaningful crowd estimates.

## Dataset and Evaluation

The project design uses the ShanghaiTech Crowd Counting Dataset as a proposed training and evaluation dataset.

The current repository does not include the dataset, trained model weights or verified accuracy metrics.

## What I'm Working on Next

- Preparing a training pipeline for ShanghaiTech
- Evaluating crowd-count predictions using MAE and RMSE
- Adding video input support
- Improving overcrowding detection for individual regions
- Testing inference performance on different hardware

## About the Project

DeepVision began as an AI/ML project exploring crowd density estimation and its potential applications in public-space monitoring.

This repository contains a starter implementation developed from the project specification. It is still under development and is not intended for real-world safety-critical monitoring.

## Author

**Dev Rai**  
B.Tech — Artificial Intelligence and Machine Learning

GitHub: https://github.com/DevCodesAI
