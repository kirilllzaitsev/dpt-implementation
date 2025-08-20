# Implementation of Vision Transformers for Dense Prediction

Reference paper: [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)

## Setup

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Experimenting

To run the training script, use:

```bash
python main.py --lr 5e-4 --epochs 200 --device cuda
```

The script will fit a model to a single sample from the [KITTI depth estimation dataset](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), stored in `sample_data/`.

## Delta with the official implementation

Official implementation can be found at https://github.com/isl-org/DPT.

Differences:

- readout token is not used
- setup of convolutional/transformer blocks (activations/dimensionality/use of normalization)