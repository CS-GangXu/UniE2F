#!/bin/bash

# Set GPU device
CUDA_VISIBLE_DEVICES=0

# Video Frame Reconstruction
python inference.py --config configs/vfr.yaml

# Video Frame Interpolation
python inference.py --config configs/vfi.yaml

# Video Frame Prediction
python inference.py --config configs/vfp.yaml