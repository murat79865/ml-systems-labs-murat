# Individual project proposal

**Student:** Murat Açıkgöz  **Student ID:** 79865
**Course:** Machine Learning (CYI5SE039CY), Group 11

## Title
Real-time traffic-sign recognition for a low-cost dashcam

## Problem
A driver-assistance dashcam must recognise the traffic sign in a cropped camera image and warn the driver
(e.g. speed limit, stop, no entry). The model runs on the device's CPU - no GPU and no cloud connection -
so it must be both accurate and small and fast enough for real-time use. Task: single-image classification
into 43 sign classes.

## Dataset
**GTSRB** - German Traffic Sign Recognition Benchmark: about 39,000 training and 12,600 test images of 43
classes, real road photos with varying light, blur and size, loaded with `torchvision.datasets.GTSRB`
(publicly available for research). Images are resized to 32 x 32 RGB. A validation split is taken from the
training set (stratified, seed 42). The classes are imbalanced (some signs are 10x more frequent than others).

## Success metric
- **Macro F1** over the 43 classes (counts rare signs equally) - target >= 0.93
- **Top-1 accuracy** on the official test set - target >= 95%
- Reported together with latency and model size; a model that breaks a constraint is not a success.

## Deployment constraints
- **Latency <= 50 ms** per image at batch 1 on a laptop CPU (median of 100 runs after warm-up)
- **Model size <= 5 MB** on disk

## Plan
1. Baseline: small CNN (3 conv blocks + linear head, about 0.3-1 M parameters, 1-4 MB in float32),
   trained with the Lab 6 pipeline (YAML config, MLflow, checkpoints), light augmentation.
2. Measure accuracy, macro F1, latency and size of the float32 baseline (Lab 1/3 methods).
3. **Planned optimization: post-training INT8 quantization** (PyTorch dynamic/static quantization) - about
   4x smaller and faster on CPU; compare with **structured pruning** of conv channels as a second option.
4. Report the accuracy / latency / size trade-off before and after optimization against the constraints.

## Risks
Class imbalance (use weighted sampling / macro F1), small 32 x 32 inputs (signs far away), accuracy drop
after quantization (fall back to quantization-aware fine-tuning).
