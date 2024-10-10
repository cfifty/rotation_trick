# Restructuring Vector Quantization with the Rotation Trick

1. **October 9, 2024:** Initial release.

This repository contains the official code for Restructuring Vector Quantization with the Rotation Trick.

![logo](assets/rot_trick_logo.jpg)

**Restructuring Vector Quantization with the Rotation Trick**  
Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iger, Jerry W. Liu, \
Ehsan Amid, Sebastian Thrun, Christopher Ré\
Under Review\
[arXiv](https://arxiv.org/abs/xxxx.yyyyy)

## Approach

In the context of VQ-VAEs, the rotation trick smoothly transforms each encoder output into its corresponding codebook
vector via a rotation and rescaling linear transformation that is treated as a constant during backpropagation. As a
result, the relative magnitude and angle between encoder output and codebook vector becomes encoded into the gradient as
it propagates through the vector quantization layer and back to the encoder.

![method](assets/rot_trick.png)

## Code environment

This code requires Pytorch 2.3.1 or higher with cuda support. It has been tested on Ubuntu 22.04.4 LTS and python 3.8.5.

You can create a conda environment with the correct dependencies using the following command lines:

```
cd rotation_trick
conda env create -f environment.yml
conda activate rotation_trick
```

## Setup

The directory structure for this project should look like:

```
Outer_Directory
│
│───rotation_trick/
│   │   src/
│   │   ...
│
│───imagenet/
│   │   train/
│   │   │   n03000134/
│   │   |   ...
│   │   val/
│   │   │   n03000247/
│   │   |   ...
```

## Training a Model

Follow the commands in ```src/scripts.sh```.

## Evaluating a Model

See ```src/reconstruction_fid.py``` as well as ```src/reconstruction_is.py``` to generate r-FID and r-IS scores
respectively from a pretrained model.
