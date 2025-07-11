# ResNet-8 Figures

## Overview
ResNet-8 Figures is a project focused on implementing and visualizing the ResNet-8 architecture, a compact version of the Residual Network (ResNet) designed for efficient image classification. This repository provides code, models, and visualizations to demonstrate the performance and structure of ResNet-8 on datasets like CIFAR-10 or Mnist.

## Features
- Implementation of ResNet-8 architecture in PyTorch.
- Training and evaluation scripts for image classification tasks.
- Visualizations of model architecture, training metrics, and sample predictions.
- Pre-trained models for quick experimentation.
- Support for CIFAR-10/Mnist dataset with data augmentation.

## Installation
### Prerequisites
- Python 3.8+
- PyTorch 1.12.0 or higher
- torchvision
- NumPy
- Matplotlib (for visualizations)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MaksKroha/resnet-8-figures.git
   cd resnet-8-figures
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training
To train the ResNet-8 model on CIFAR-10:
```bash
python train.py --dataset cifar10 --epochs 100 --batch-size 128
```
- Use `--help` for additional arguments like learning rate or model save path.

### Evaluation
To evaluate a pre-trained model:
```bash
python evaluate.py --model-path models/resnet8_cifar10.pth
```

### Visualizations
Generate visualizations of the model architecture and training metrics:
```bash
python visualize.py --model-path models/resnet8_cifar10.pth --output-dir figures/
```

## Project Structure
```
resnet-8-figures/
├── models/               # Pre-trained models
├── figures/              # Generated visualizations
├── src/                  # Source code
│   ├── model.py          # ResNet-8 architecture
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── visualize.py      # Visualization script
├── data/                 # Dataset loading utilities
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Results
- Achieves ~90% accuracy on CIFAR-10 test set after 100 epochs.
- Visualizations include loss/accuracy curves, confusion matrices, and sample predictions.
- Model size: ~0.5M parameters, optimized for low-resource environments.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, feel free to open an issue or contact [MaksKroha](https://github.com/MaksKroha).
