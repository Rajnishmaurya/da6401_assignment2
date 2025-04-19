# ResNet50 Fine-Tuning last K layers

This repository contains code for fine-tuning a pre-trained ResNet50 model on the iNaturalist dataset. The project demonstrates transfer learning by freezing most of the ResNet50 layers and only training the last k layers for image classification.

## Project Overview

The code fine-tunes a pre-trained ResNet50 model on the iNaturalist_12K dataset, which contains nature images across 10 classes. The approach involves:

1. Loading a pre-trained ResNet50 model
2. Freezing most of the model layers
3. Unfreezing and training only the last k layers
4. Tracking experiments with Weights & Biases

## Dataset

The project uses the iNaturalist_12K dataset, which should be organized as follows:
```
/kaggle/inpuut/nature/inaturalist_12K
  /train  - Training images organized in class folders
  /val    - Validation/test images organized in class folders
```

## Requirements

- PyTorch
- torchvision
- Weights & Biases (wandb)
- CUDA-compatible GPU (recommended)

## Installation

```bash
pip install torch torchvision wandb
```

## Usage

1. Set up your dataset paths in the script
2. Configure hyperparameters via Weights & Biases config
3. Run the training script:

```bash
python train.py
```

## Configuration

The main configuration parameters are:
- `batch_size`: 32
- `epochs`: 10
- `learning_rate`: 1e-4
- `k`: 20 (number of layers to unfreeze)

These can be modified through the wandb config object.

## Model Architecture

The implementation:
1. Loads a pre-trained ResNet50 model
2. Replaces the final fully connected layer to match the 10 classes in the dataset
3. Freezes all initial layers
4. Unfreezes only the last k(20) layers for fine-tuning

## Training Process

The training loop:
1. Trains using the specified number of epochs
2. Evaluates on a validation set after each epoch
3. Logs metrics (loss, accuracy) to Weights & Biases
4. Performs final evaluation on the test set

## Monitoring

Training progress is monitored through Weights & Biases, tracking:
- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Final test accuracy

## Results

After training, the model's performance will be displayed as final test accuracy and logged to the Weights & Biases dashboard.


## Links
[Wandb Report](https://api.wandb.ai/links/da24m015-iitm/4rain58f)
[Github](https://github.com/Rajnishmaurya/da6401_assignment2/tree/main/partB)