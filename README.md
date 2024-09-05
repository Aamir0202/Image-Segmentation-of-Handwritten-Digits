# Handwritten Digit Segmentation with CNN and FCN-8

## Overview

This project focuses on building a Convolutional Neural Network (CNN) from scratch to predict pixel-wise segmentation masks for handwritten digits. The model is trained on the M2NIST dataset, which extends the MNIST dataset to include multiple digits in a single image. The architecture leverages a Fully Convolutional Network (FCN-8) for upsampling to produce precise segmentation maps.

## Project Structure

- `data/`: Contains the M2NIST dataset and any preprocessing scripts.
- `models/`: Includes the implementation of the CNN and FCN-8 architectures.
- `notebooks/`: Jupyter notebooks for exploratory analysis and model training.
- `scripts/`: Python scripts for training, evaluation, and inference.
- `README.md`: This file.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```
### Model Architecture

- **CNN Downsampling Path**: Custom-built CNN layers for feature extraction.
- **FCN-8 Upsampling Path**: Utilizes an FCN-8 architecture for upsampling and generating pixel-wise segmentation masks.

### Evaluation

Evaluate the model using Intersection over Union (IOU) and Dice Score metrics. To run the evaluation, use:

## Contributing

Feel free to fork this repository and submit pull requests with improvements or fixes. Ensure that you follow the coding guidelines and include tests for any new features.

## Acknowledgements

- [M2NIST dataset](dataset_source/link)
- [Fully Convolutional Networks (FCN)](https://arxiv.org/abs/1605.06211)
- [TensorFlow Documentation](https://www.tensorflow.org/)
