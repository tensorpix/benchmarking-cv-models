<p align="center" >
  <img width="400" src="https://cdn.tensorpix.ai/TensorPix-Logo-color.svg" alt="Tensorpix logo"/>
</p>

---

# Benchmarking CV models

Docker image for simple training benchmark of popular computer vision models.

The benchmark code explicitly focuses on benchmarking only the **pure training loop code**. The dataset is
generated on the fly and directly in RAM with minimal overhead.

There is no extra work done in the training loop such as data preprocessing, model saving, validation, logging...

We use [Lightning AI](https://lightning.ai/) library for benchmarks as it's a popular tool among deep learning practitioners.

It also supports features such as mixed precision, DDP, and multi-GPU training.
Such features can significantly affect benchmark performance so it's important to offer them in benchmarks.

## ❓ Why did we create this?

[Our](https://tensorpix.ai) ML team had a dilemma while choosing the best GPU for our budget. GPU X was 2x the price of GPU Y, but we couldn't find reliable data that shows if GPU X was also 2x the speed of GPU Y.

There were [some benchmarks](https://lambdalabs.com/gpu-benchmarks), but very few of them were specific for computer vision tasks and even fewer for the GPUs we wanted to test. So we created a docker image that does this with minimal setup.

You can use this benchmark repo to:

- See how various GPUs perform on various deep CV architectures
- Benchmark various CV architectures
- See how efficient are multi-GPU setups for a specific GPU
- Test how much you gain in training speed when using Mixed-precision
- Stress test the GPU(s) at near 100% utilization
- Make pizzas (not tested)

## 📋 Supported architectures

Please open an issue if you need support for a new architecture.

- ResNet50
- ConvNext (base)
- VGG16
- Efficient Net v2
- MobileNet V3
- ResNeXt50
- SWIN
- VIT
- UNet with ResNet50 backbone

## 📖 How to benchmark

### Prerequisites

In order to run benchmark docker containers you must have the following installed on the host machine:

- Docker (we used v24.0.6 for testing)
- NVIDIA drivers. See [Versions](#versions) when choosing the docker image.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) - required in order to use CUDA inside docker containers

### Training vs Inference

By default, the container will benchmark model training. If you want to benchmark model inference, append the `src.inference` to the docker run command. See examples below for more details.

### Examples

**Minimal**

`docker run --rm --ipc=host --ulimit memlock=-1 --gpus all ghcr.io/tensorpix/benchmarking-cv-models src.train --batch-size 32`

**Advanced**

`docker run --rm --ipc=host --ulimit memlock=-1 --gpus '"device=0,1"' -v ./benchmarks:/workdir/benchmarks ghcr.io/tensorpix/benchmarking-cv-models src.train --batch-size 32 --n-iters 1000 --warmup-steps 100 --model resnext50 --precision 16-mixed --width 320 --height 320`

**Benchmark Inference**

`docker run --rm --ipc=host --ulimit memlock=-1 --gpus all ghcr.io/tensorpix/benchmarking-cv-models src.inference --batch-size 32 --n-iters 1000 --model resnext50 --precision 16 --width 256 --height 256`

**List all train options:**

`docker run --rm ghcr.io/tensorpix/benchmarking-cv-models src.train --help`

**List all inference options:**

`docker run --rm ghcr.io/tensorpix/benchmarking-cv-models src.inference --help`

### How to select particular GPUs

If you want to use all available GPUs, then set the `--gpus all` docker parameter.

If want to use for example GPUs at indicies 2 and 3, set `--gpus '"device=2,3"'`.

### Logging results to a persistent CSV file

Benchmark code will create a CSV file with benchmark results on every run. The file will exist inside the docker container, but you have to mount it in order to see it on the host machine.

To do so, use the following docker argument when running a container: `-v <host/benchmark/folder>:/workdir/benchmarks`. See the [advanced example](#examples) for more details. The CSV file will reside in the mounted host directory.

We also recommend that you create the `<host/benchmark/folder>` on the host before running the container as the container will create the folder under the `root` user if it doesn't exist on the host.

### Versions

We support two docker images: one for CUDA 12.0 and second for CUDA 11.8. The `12.0` version is on the latest docker tag, while `11.8` is on the `ghcr.io/tensorpix/benchmarking-cv-models:cuda118` tag.

`11.8` version supports earlier NVIDIA drivers so if you run into driver related errors, try this image instead.

## 📊 Metrics

We use 3 metrics for the benchmark:

- Images per second
- Batches per second
- Megapixels per second

Images/s and batches/s are self-explanatory. Megapixels/s (MPx) are not usually used but we like this metric as it's input resolution independent.

It's calculated according to the following formula: `(input_width_px * input_height_px * batch_size * n_gpus * n_iterations) / (elapsed_time_s * 10^6)`
