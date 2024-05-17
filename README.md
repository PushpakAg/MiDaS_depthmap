# Real-Time Depth Estimation using MiDaS

Python script that captures real-time video from a webcam and generates a corresponding depth map for each frame using the MiDaS (Monocular Depth Sensing) model. The script processes each frame to predict depth information and overlays the frames with FPS (frames per second) information. The original image and the depth map are then displayed side-by-side in real time.

## Requirements

Make sure you have the following dependencies installed:
- OpenCV
- PyTorch
- NumPy

You can install these dependencies using pip:

```bash
pip install opencv-python torch numpy
```
Clone this repository:

```bash
git clone https://github.com/PushpakAg/MiDaS_depthmap.git
cd MiDaS_depthmap
```

The script supports three model types for depth estimation:

- DPT_Large (highest accuracy, slowest inference speed)
- DPT_Hybrid (medium accuracy, medium inference speed)
- MiDaS_small (lowest accuracy, highest inference speed)

```bash
# Available options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
model_type = "MiDaS_small"
```


https://github.com/PushpakAg/MiDaS_depthmap/assets/36948404/c8aae0cb-1702-4e69-95dc-342fd2b8b354





