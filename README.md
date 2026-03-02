# Event-based Novel View Synthesis using Stable Video Diffusion

This repository contains the official implementation of our paper on event-based novel view synthesis using Stable Video Diffusion (SVD).

[video demo.webm](https://github.com/user-attachments/assets/76eddfe0-8c3c-4b18-b24a-eab950ff3763)

## Environment Setup

```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset should be organized as follows:

```
data/
├── train/
│   ├── scene_001/
│   │   ├── rgb/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── event_corrected/
│   │       ├── 000001.npz
│   │       ├── 000002.npz
│   │       └── ...
│   ├── scene_002/
│   └── ...
└── test/
    ├── scene_001/
    └── ...
```

### Data Format Details

- **RGB images**: JPG format, resolution 448x320
- **Event data**: NPZ format with key `data`, shape (3, H, W), values in range [-N, N]

## Pretrained Models

Please download the following pretrained models and place them in the `pretrained/` directory:

1. **vae_event.pth**: VAE model for event encoding
2. **resnet.pth**: ResNet model for feature extraction

After training, the UNet checkpoint will be saved to `checkpoints/unet_train_params.pth`.

## Usage

### Training

```bash
python train.py --config configs/unie2f.yaml
```

### Inference

```bash
python inference.py --config configs/unie2f.yaml
```

### Evaluation

```bash
python evaluation.py --config configs/unie2f.yaml
```

Or run all steps with:

```bash
bash run.sh
```

## Configuration

Edit `configs/unie2f.yaml` to modify:
- Data paths
- Training parameters
- Inference settings

## License

This project is licensed under the Apache License 2.0.

## Acknowledgments

This code is built upon the HuggingFace diffusers library and Stable Video Diffusion.
