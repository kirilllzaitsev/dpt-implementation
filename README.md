# Implementation of Vision Transformers for Dense Prediction

Reference paper: [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)

## Delta with the official implementation

Official implementation can be found at https://github.com/isl-org/DPT.

This implementation supports the following backbones:

- [x]  ResNet50

- [ ]  ViT-Base

Also:

- readout token is not used
- there are minor differences  in convolutional blocks (activations/dimensionality/use of normalization)