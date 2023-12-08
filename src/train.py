import argparse

import segmentation_models_pytorch as smp
import torch
from lightning import Trainer
from pip._internal.operations import freeze
from torch.utils.data import DataLoader
from torchvision.models import (
    convnext_base,
    efficientnet_v2_m,
    mobilenet_v3_large,
    resnet50,
    resnext50_32x4d,
    swin_b,
    vgg16,
    vit_b_16,
)

from src import log
from src.callbacks import BenchmarkCallback
from src.data.in_memory_dataset import InMemoryDataset
from src.models.lightning_modules import LitClassification

logger = log.setup_custom_logger()

ARCHITECTURES = {
    "resnet50": resnet50,
    "convnext": convnext_base,
    "vgg16": vgg16,
    "efficient_net_v2": efficientnet_v2_m,
    "mobilenet_v3": mobilenet_v3_large,
    "resnext50": resnext50_32x4d,
    "swin": swin_b,
    "vit": vit_b_16,
    "unet_resnet50": smp.Unet
    # TODO"ssd_vgg16": ssd300_vgg16,
    # TODO "fasterrcnn_resnet50_v2": fasterrcnn_resnet50_fpn_v2,
}


def print_requirements():
    pkgs = freeze.freeze()
    for pkg in pkgs:
        logger.info(pkg)


def main(args):
    if args.list_requirements:
        print_requirements()

    args_dict = vars(args)
    logger.info(f"User Arguments {args_dict}")

    dataset = InMemoryDataset(width=args.width, height=args.width)
    data_loader = DataLoader(
        dataset,
        num_workers=args.n_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        strategy="ddp",
        precision=args.precision,
        limit_train_batches=args.n_iters + args.warmup_steps,
        max_epochs=1,
        callbacks=[
            BenchmarkCallback(
                warmup_steps=args.warmup_steps,
                model_name=args.model,
                precision=args.precision,
                workers=args.n_workers,
            )
        ],
        devices=args.devices,
    )

    if args.model in ARCHITECTURES:
        if args.model == "unet_resnet50":
            model = ARCHITECTURES[args.model](
                encoder_name="resnet50", encoder_weights=None
            )
        else:
            model = ARCHITECTURES[args.model]()

    else:
        raise ValueError("Architecture not supported.")

    model = LitClassification(model=model)
    trainer.fit(model=model, train_dataloaders=data_loader)


if __name__ == "__main__":
    logger.info("########## STARTING NEW BENCHMARK RUN ###########")

    if not torch.cuda.is_available():
        raise ValueError("CUDA device not found on this system.")
    else:
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(
            f"CUDA Device Total Memory: {(torch.cuda.get_device_properties(0).total_memory / 1e9):.2f} GB"
        )

    parser = argparse.ArgumentParser(description="Benchmark CV models training on GPU.")

    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--n-iters",
        type=int,
        default=300,
        help="Number of training iterations to benchmark for. One iteration = one batch update",
    )
    parser.add_argument(
        "--precision", choices=["32", "16", "16-mixed", "bf16-mixed"], default="32"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of Data Loader workers. CPU shouldn't be a bottleneck with 4+.",
    )
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--width", type=int, default=224, help="Input width")
    parser.add_argument("--height", type=int, default=224, help="Input height")

    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--accelerator", choices=["gpu"], default="gpu", help="Accelerator to use."
    )
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

    if args.warmup_steps <= 0:
        raise ValueError("Number of warmup steps must be > 0")

    main(args=args)
