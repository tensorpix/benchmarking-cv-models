import argparse
from pprint import pprint

import pkg_resources
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models import (
    convnext_base,
    efficientnet_v2_m,
    fasterrcnn_resnet50_fpn_v2,
    mobilenet_v3_large,
    resnet50,
    resnext50_32x4d,
    ssd300_vgg16,
    swin_b,
    vgg16,
    vit_b_16,
)

from src.callbacks import BenchmarkCallback
from src.data.in_memory_dataset import InMemoryDataset
from src.models.lightning_modules import LitClassification

ARCHITECTURES = {
    "resnet50": resnet50,
    "convnext": convnext_base,
    "vgg16": vgg16,
    "efficient_net_v2": efficientnet_v2_m,
    "mobilenet_v3": mobilenet_v3_large,
    "resnext50": resnext50_32x4d,
    "swin": swin_b,
    "vit": vit_b_16,
    "ssd_vgg16": ssd300_vgg16,
    "fasterrcnn_resnet50_v2": fasterrcnn_resnet50_fpn_v2,
}


def print_requirements():
    env = dict(tuple(str(ws).split()) for ws in pkg_resources.working_set)
    for k, v in env.items():
        print(f"{k}=={v}")


def main(args):
    if args.list_requirements:
        print_requirements()
        print()

    args_dict = vars(args)
    print("Arguments:")
    pprint(args_dict)

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
        max_steps=args.n_iters + args.warmup_steps,
        max_epochs=1,
        callbacks=[BenchmarkCallback(warmup_steps=args.warmup_steps)],
        devices=args.devices,
    )

    if args.model.lower in ARCHITECTURES:
        model = ARCHITECTURES[args.model.lower]()
    else:
        raise ValueError("Architecture not supported.")

    model = LitClassification(model=model)
    trainer.fit(model=model, train_dataloaders=data_loader)


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
        default=300,
        help="Number of training iterations to benchmark for. One iteration = one batch update",
    )
    parser.add_argument(
        "--precision", choices=["32", "16", "16-mixed", "bf16-mixed"], default="32"
    )
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--width", type=int, default=192, help="Input width")
    parser.add_argument("--height", type=int, default=192, help="Input height")

    parser.add_argument("--warmup-steps", type=int, default=50)
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
