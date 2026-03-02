#!/bin/bash

# Set GPU device
CUDA_VISIBLE_DEVICES=0

# Train
python train.py --config configs/base.yaml

# Inference
python inference.py --config configs/vfr.yaml

# Evaluation
python evaluation.py --config configs/vfr.yaml