# ED-PLL

This is the code for the paper: Evidential Deep Partial Label Learning to Quantify Disambiguation Uncertainty


To be presented at CVPR 2026.

## Setups

All code was developed and tested on a single machine equiped with a NVIDIA Tesla V100 GPU. The environment is as bellow:
- Python 3.6.8
- Numpy 1.16.4
- Cuda 10.1.168

## Quick Start

Here is an example:

```
python main.py --dataset mnist --model linear --partial_type binomial --partial_rate 0.3
```
