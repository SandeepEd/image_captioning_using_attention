# Image Captioning with and without Attention

An image captioning model trained using attention mechanisms on the Flickr8k dataset.  This project implements and compares multiple deep learning architectures for generating descriptive captions from images.

## Overview

This project demonstrates how attention mechanisms can improve image captioning by allowing the model to focus on different regions of an image when generating each word of the caption albeit the training on the resnet for image features was frozen but the model could learn better if the resnet was trained by unfreezing the few last layers. The repository includes implementations of:

- **CNN-LSTM with Attention**:  Combines convolutional neural networks for image feature extraction with LSTM units enhanced by attention mechanisms
- **CNN-LSTM without Attention**: A baseline model for performance comparison
- **Transformer Architecture**: A modern attention-based approach to image captioning

## Dataset

The model is trained on the **Flickr8k dataset**, which contains 8,000 images with 5 captions per image, providing rich annotations for training and evaluation. But the transformer architecture can be trained on larger datasets like Flickr30k or MS COCO for better performance.

## Project Structure

- `main.ipynb` - Complete pipeline and experimentation notebook
- `cnn_lstm_with_attention.py` - CNN-LSTM model with attention mechanism
- `cnn_lstm_without_att.py` - Baseline CNN-LSTM model without attention
- `transformer_arc.py` - Transformer-based image captioning architecture
- `utils.py` - Utility functions for data processing and evaluation

## Key Features

- Multiple architecture implementations for comparison
- Comprehensive training pipeline with evaluation metrics

## Getting Started

1. Clone the repository
2. Install dependencies (PyTorch, torchvision, numpy, etc.)
3. Download the Flickr8k dataset
4. Run `main.ipynb` to train and evaluate models, might need some editing if you are using this codebase for your own experiments

## Requirements

- Python 3.7+
- PyTorch
- TorchVision
- NumPy
- Jupyter Notebook

## License

This project is provided as-is for educational and research purposes.