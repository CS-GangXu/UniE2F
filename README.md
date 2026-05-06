# UniE2F: A Unified Diffusion Framework for Event-to-Frame Reconstruction with Video Foundation Models

This repository contains the official implementation of our paper "UniE2F: A Unified Diffusion Framework for Event-to-Frame Reconstruction with Video Foundation Models" on event-based novel view synthesis using Stable Video Diffusion (SVD).

[video demo.webm](https://github.com/user-attachments/assets/76eddfe0-8c3c-4b18-b24a-eab950ff3763)

## Environment Setup

Please install the diffusers by:
```bash
pip install diffusers["torch"] transformers
pip install accelerate
pip install -e ".[torch]"
```

Please install required packages by:
```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset should be organized as follows:

```
data/
├── scene_001/
│   ├── rgb/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── event_corrected/
│       ├── 000001.npz
│       ├── 000002.npz
│       └── ...
├── scene_002/
└── ...
```

### Data Format Details

- **RGB images**: JPG format, resolution 448x320
- **Event data**: NPZ format with key `data`, shape (3, H, W), values in range [-N, N]

## Pretrained Models

Please download the [pretrained models](https://pan.baidu.com/s/1HfE0EMj0kcjYcepDV_H6hw?pwd=kwix) and place them in the `parameter` directory:

1. **vae_event.pth**: VAE model for event encoding
2. **resnet.pth**: ResNet model for residual prediction
3. **unet.pth**: U-Net model for denoising

## Inference

```bash
# Video Frame Reconstruction
python inference.py --config exp/vfr.yaml

# Video Frame Interpolation
python inference.py --config exp/vfi.yaml

# Video Frame Prediction
python inference.py --config exp/vfp.yaml
```

## License

This project is licensed under the Apache License 2.0.

## Acknowledgments

This code is built upon the HuggingFace diffusers library and Stable Video Diffusion.
