# Benchmarking CV models

Docker image for simple training benchmark of popular computer vision models.

The benchmark code explicitly focuses on benchmarking only the **pure training loop code**. The dataset is
generated on the fly and directly in RAM with minimal overhead.

There is no extra work done in the training loop such as data preprocessing, model saving, validation, logging...

We use [Lightning AI](https://lightning.ai/) library for benchmarks as it's a popular tool among deep learning practitioners.

It also supports features such as mixed precision, DDP, and multi-GPU training.
Such features can significantly affect benchmark performance so it's important to offer them in benchmarks.

## Why did we create this?

[Our](https://tensorpix.ai) ML team had a dilemma while choosing the best GPU for our budget. GPU X was 2x the price of GPU Y, but we couldn't find reliable data that shows if GPU X was also 2x the speed of GPU Y.

There were some benchmarks, but very few of them were specific for computer vision tasks. So... we created our own mini-library that does this.

You can use this benchmark repo to:

- See how various GPUs perform on various deep CV architectures
- Benchmark various CV architectures
- See how efficient are multi-GPU setups for a specific GPU
- Test how much you gain in training speed when using Mixed-precision
- Make pizzas (not tested)

## How to benchmark

### Prerequisites

In order to run benchmark docker containers you must have the following installed on the host machine:

- Docker (we used v24.0.6 for testing)
- NVIDIA drivers. See [Versions](#versions) when choosing the docker image.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) - required in order to use CUDA inside docker containers

### Examples

**Minimal**

`docker run --rm --ipc=host --ulimit memlock=-1 --gpus all ghcr.io/tensorpix/benchmarking-cv-models --batch-size 32`

**Advanced**

`docker run --rm --ipc=host --ulimit memlock=-1 --gpus all -v ./benchmarks:/workdir/benchmarks ghcr.io/tensorpix/benchmarking-cv-models --batch-size 32 --n-iters 1000 --warmup-steps 100 --model resnext50 --precision 16-mixed --width 320 --height 320 --devices 3`

**List all options:**

`docker run --rm ghcr.io/tensorpix/benchmarking-cv-models --help`

### Logging results to a persistent CSV file

Benchmark code will create a CSV file with benchmark results on every run. The file will exist inside the docker container, but you have to mount it in order to see it on the host machine.

To do so, use the following docker argument when running a container: `-v <host/benchmark/folder>:/workdir/benchmarks`. See the [advanced example](#examples) for more details. The CSV file will reside in the mounted host directory.

### Versions

We support two docker images: one for CUDA 12.0 and second for CUDA 11.8. The `12.0` version is on the latest docker tag, while `11.8` is on the `ghcr.io/tensorpix/benchmarking-cv-models:cuda118` tag.

`11.8` version supports earlier NVIDIA drivers so if you run into driver related errors, try this image instead.
