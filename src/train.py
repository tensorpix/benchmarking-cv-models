import argparse
from pprint import pprint

import pkg_resources
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models import (
    alexnet,
    convnext_base,
    efficientnet_v2_m,
    mobilenet_v3_large,
    resnet50,
    resnext50_32x4d,
    swin_b,
    vgg16,
    vit_b_16,
)

from src.callbacks import BenchmarkCallback
from src.data.in_memory_dataset import InMemoryDataset
from src.models.lightning_modules import LitClassification


def print_requirements():
    env = dict(tuple(str(ws).split()) for ws in pkg_resources.working_set)
    for k, v in env.items():
        print(f"{k}=={v}")


def choose_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return resnet50()
    elif model_name == "convnext":
        return convnext_base()
    elif model_name == "vgg16":
        return vgg16()
    elif model_name == "alexnet":
        return alexnet()
    elif model_name == "efficient_net_v2":
        return efficientnet_v2_m()
    elif model_name == "mobilenet_v3":
        return mobilenet_v3_large()
    elif model_name == "resnext_50":
        return resnext50_32x4d()
    elif model_name == "swin":
        return swin_b()
    elif model_name == "vit":
        return vit_b_16()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


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
        accelerator="gpu",
        strategy="ddp",
        precision=args.precision,
        max_steps=args.n_iters + args.warmup_steps,
        max_epochs=1,
        callbacks=[BenchmarkCallback(warmup_steps=args.warmup_steps)],
    )

    model = choose_model(args.model)
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
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--n-workers", type=int, default=4)

    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)

    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--list-requirements", action="store_true")

    args = parser.parse_args()
    main(args=args)
