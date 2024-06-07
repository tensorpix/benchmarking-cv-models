import argparse
from pprint import pprint

import torch
import torch.utils.benchmark as benchmark

ARCHITECTURES = {
    "resnet50": "resnet50",
    "convnext": "convnext_base",
    "vgg16": "vgg16",
    "efficient_net_v2": "efficientnet_v2_m",
    "mobilenet_v3": "mobilenet_v3_large",
    "resnext50": "resnext50_32x4d",
    "swin": "swin_b",
    "vit": "vit_b_16",
    "ssd_vgg16": "ssd300_vgg16",
    "fasterrcnn_resnet50_v2": "fasterrcnn_resnet50_fpn_v2",
}


def benchmark_inference(
    stmt: str,
    setup: str,
    input: torch.Tensor,
    n_runs=100,
    num_threads: int = 1,
):
    """
    Benchmark a model using torch.utils.benchmark.

    When evaluating model speed in MP/s only the video height, width and batch size are taken into
    account. The number of channels and sequence length are ignored. Speed evaluation measures
    how fast can we process an arbitrary input video so channels and sequence length don't
    affect the model computation speed.
    """

    timer = benchmark.Timer(
        stmt=stmt,
        setup=setup,
        num_threads=num_threads,
        globals={"x": input},
    )

    print(
        f"Running benchmark on sample of {n_runs} runs with {num_threads} thread(s)..."
    )
    result = timer.timeit(n_runs)

    batch, height, width = input.size(0), input.size(-2), input.size(-1)
    total_pixels = batch * width * height

    print(f"Batch size: {batch}")
    print(f"Input resolution: {width}x{height} pixels\n")

    mean_per_batch = result.mean
    median_per_batch = result.median

    mean_speed_mpx = (total_pixels / 1e6) / mean_per_batch
    median_speed_mpx = (total_pixels / 1e6) / median_per_batch

    print(f"Mean time per {batch} {width}x{height} px frames: {mean_per_batch:.4f} s")
    print(
        f"Median time per {batch} {width}x{height} px frames: {median_per_batch:.4f} s\n"
    )

    print(
        f"Model mean throughoutput in megapixels per second: {mean_speed_mpx:.3f} MP/s"
    )
    print(
        f"Model median throughoutput in megapixels per second: {median_speed_mpx:.3f} MP/s\n"
    )


def main(args):
    args_dict = vars(args)
    print("Arguments:")
    pprint(args_dict)

    if args.model.lower() not in ARCHITECTURES:
        raise ValueError("Architecture not supported.")

    stmt = """ \
    with torch.inference_mode():
        out = model(x)
        out = out.clamp(0, 1).float().cpu()
    """

    arch = ARCHITECTURES[args.model.lower()]
    setup = f"from torchvision.models import {arch}; model = {arch}(); model.eval()"

    input_shape = [args.batch_size, 3, args.height, args.width]
    precision = torch.float16 if args.precision == "16" else torch.float32

    x = torch.rand(*input_shape, dtype=precision)
    x = x.cuda(0, non_blocking=True)
    setup = f"{setup}; model.cuda(0)"

    if args.precision == "16":
        setup = f"{setup}; model.half()"

    benchmark_inference(
        stmt=stmt,
        setup=setup,
        input=x,
        n_runs=args.n_iters,
        num_threads=args.n_workers,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise ValueError("CUDA device not found on this system.")
    else:
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("CUDNN version:", torch.backends.cudnn.version())
        print(
            "CUDA Device Total Memory: "
            + f"{(torch.cuda.get_device_properties(0).total_memory / 1e9):.2f} GB",
        )

    parser = argparse.ArgumentParser(description="Benchmark CV models training on GPU.")

    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help="Number of training iterations to benchmark for. One iteration = one batch update",
    )
    parser.add_argument("--precision", choices=["32", "16"], default="16")
    parser.add_argument("--n-workers", type=int, default=1)

    parser.add_argument("--width", type=int, default=192, help="Input width")
    parser.add_argument("--height", type=int, default=192, help="Input height")

    parser.add_argument(
        "--model",
        default="resnet50",
        choices=list(ARCHITECTURES.keys()),
        help="Architecture to benchmark.",
    )
    parser.add_argument("--list-requirements", action="store_true")

    args = parser.parse_args()

    if args.n_iters <= 0:
        raise ValueError("Number of iterations must be > 0")

    main(args=args)
