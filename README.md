# Real-Time Depth Estimation using MiDaS

This repository contains a Python script that captures real-time video from a webcam and generates a corresponding depth map for each frame using the MiDaS (Monocular Depth Sensing) model. The script processes each frame to predict depth information and overlays the frames with FPS (frames per second) information. The original image and the depth map are then displayed side-by-side in real time.

## Requirements

Make sure you have the following dependencies installed:
- OpenCV
- PyTorch
- NumPy

You can install these dependencies using pip:

```bash
pip install opencv-python torch numpy
