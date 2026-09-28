# AD-2 Restormer

A compact PyTorch image-restoration project for paired adverse-weather image data. It combines a lightweight encoder-decoder restoration network with an AD-2 degradation-conditioning branch.

## What It Demonstrates

- Paired-image dataset loading and preprocessing
- PyTorch model implementation
- Train/validation/test dataset splitting
- GPU-aware training
- Checkpoint-based inference
- Degradation-conditioned feature processing

## Architecture

`SimpleRestormer` has two main components:

1. **AD2Module** extracts global image features and predicts a two-class degradation distribution.
2. **Restoration backbone** encodes the image, conditions intermediate features with the AD-2 branch, and reconstructs a restored RGB image.

The forward pass returns:

```python
restored_image, degradation_prediction = model(x)
```

## Repository Structure

```text
dataset/
  full/
  train/
  val/
  test/
dataset.py
model.py
split.py
train.py
test.py
model.pth
```

Current bundled split:

| Split | Image pairs |
| --- | ---: |
| Train | 14,455 |
| Validation | 1,807 |
| Test | 1,807 |

## Requirements

```bash
pip install torch opencv-python numpy tqdm
```

## Training

```bash
python train.py
```

The current training script:

- Detects CUDA automatically
- Uses 256×256 paired images
- Uses Adam with learning rate `1e-4`
- Uses L1 loss
- Saves weights to `model.pth`

## Inference

```bash
python test.py
```

The current inference script loads `model.pth`, processes a sample image, and writes the restored result to `output.png`.

## Dataset Preparation

```bash
python split.py
```

The split script creates an 80/10/10 train-validation-test partition from matching files in `dataset/full/input` and `dataset/full/target`.

## Current Limitations

- Fixed 256×256 image size
- Hardcoded training and inference paths
- Training defaults to one epoch
- No CLI or configuration file
- No checkpoint versioning/resume flow
- No PSNR/SSIM reporting yet
- The degradation branch is computed, while the current loss optimizes restoration output only
- Dataset splitting is not seeded, so reruns are not deterministic

## Next Improvements

- Add CLI arguments and configuration
- Add deterministic dataset splitting
- Add PSNR and SSIM validation metrics
- Add checkpoint versioning and resume support
- Support arbitrary inference paths
- Add experiment logging

## Status

Research/learning prototype focused on understanding image restoration pipelines and degradation-conditioned neural networks.
