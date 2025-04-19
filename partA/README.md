# CNN Image Classification Project

This project implements a Convolutional Neural Network (CNN) for image classification on the iNaturalist dataset. It includes hyperparameter optimization using Weights & Biases (wandb) and model evaluation.

## Project Structure

- **Training Pipeline**: Implements a CNN architecture with configurable hyperparameters
- **Hyperparameter Optimization**: Uses Bayesian optimization to find the best model configuration
- **Evaluation**: Tests the model on validation data and visualizes predictions

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- wandb
- matplotlib
- numpy
- Pillow (PIL)
- CUDA-capable GPU (recommended)

## Installation

```bash
pip install torch torchvision wandb matplotlib numpy pillow
```

## Dataset

The code expects the iNaturalist_12K dataset with the following structure:
```
/kaggle/input/nature/inaturalist_12K/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Usage

### Training with Hyperparameter Optimization

```python
# Initialize wandb
wandb.login()

# Run the hyperparameter sweep
sweep_id = wandb.sweep(
    sweep=hyperparameter_space,
    project="assignment2_partA"
)
wandb.agent(sweep_id, function=hyperparameter_optimization, count=10)
```

### Hyperparameters

The model explores the following hyperparameters:
- Mini batch size: [32, 64]
- Channel architecture: [[32,32,32,32,32], [32,64,128,256,512], [128,64,32,16,8]]
- Normalization: [True, False]
- Activation function: ["ReLU", "GELU", "SiLU"]
- Hidden layer size: [128, 256, 512]
- Dropout probability: [0.2, 0.3]
- Data augmentation: [True, False]
- Learning rate: [0.001, 0.002, 0.005]

### Model Architecture

The `NeuralImageClassifier` class implements a flexible CNN with:
- Configurable convolutional feature extraction layers
- Optional batch normalization
- Selectable activation functions
- Fully-connected classification head with dropout

### Evaluation

To evaluate a trained model:

```python
# Load best hyperparameters
with open("best_hyperparameters.json", "r") as f:
    config = json.load(f)

# Rebuild model with best configuration
model = NeuralImageClassifier(
    channel_sizes=config["channel_architecture"],
    activation_function=getattr(nn, config["nonlinearity"]),
    hidden_size=config["hidden_layer_size"],
    dropout_prob=config["dropout_probability"],
    use_normalization=config["use_normalization"],
    class_count=10,
    image_dimensions=224
).to(device)

# Load weights
model.load_state_dict(torch.load("best_model.pth"))

# Run evaluation
# ... see evaluation code in the script
```

## Performance Optimization

The implementation includes:
- Mixed precision training with `torch.amp`
- Gradient scaling for numerical stability
- Gradient clipping to prevent exploding gradients
- Efficient data loading with DataLoader workers and pin_memory

## Visualization

The evaluation script generates visualization of model predictions for 3 random samples from each class, displaying both true and predicted labels.

## Results

After training, the best model configuration is saved to:
- `best_model.pth`: Trained model weights
- `best_hyperparameters.json`: Best hyperparameter configuration

## Links
[Wandb Report](https://api.wandb.ai/links/da24m015-iitm/4rain58f)  
[Github](https://github.com/Rajnishmaurya/da6401_assignment2/tree/main/partA)