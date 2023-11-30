# Benchmarking CV models

Docker image for simple training benchmark of popular computer vision models.

The benchmark code explicitly focuses on benchmarking only the **pure training loop code**. The dataset is
generated on the fly and directly in RAM with minimal overhead.

There is no extra work done in the training loop such as data preprocessing, model saving, validation, logging...

We use [Lightning AI](https://lightning.ai/) library for benchmarks as it's a popular tool among deep learning practitioners.
It also supports important features such as mixed precision, DDP, and multi-GPU training...
Such features can significantly affect benchmark performance so it's important to offer them in benchmarks.

## Development

Prepare Pre-commit hooks: `pre-commit install`

## Building image

### Prerequisites

Your host system must have an NVIDIA driver version 525 or higher installed.

### Steps

0. **OPTIONAL:** Choose the base image in `Dockerfile`. The image must have CUDA installed.
1. Build `docker build -t cv-benchmark .`

## How to benchmark

### Minimal example

`docker run --ipc=host --ulimit memlock=-1 --gpus all cv-benchmark --batch-size 32`

### Advanced example

`docker run --ipc=host --ulimit memlock=-1 --gpus all cv-benchmark --batch-size 32 --n-iters 1000 --warmup-steps 100 --model resnext50 --precision 16-mixed --width 320 --height 320 --devices 3`

### List all options

`docker run cv-benchmark --help`
