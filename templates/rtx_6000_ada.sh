#!/bin/bash
# GPU: 1x NVIDIA RTX 6000 Ada, 48 GB VRAM

#cfg32=("resnet50,644" "resnext50,512" "unet_resnet50,440" "swin,240" "convnext,256")
cfg16=("resnet50,1280" "resnext50,1024" "unet_resnet50,880" "swin,360" "convnext,500")

N_ITERS=300
PRECISION="16-mixed"

for str in ${cfg16[@]}; do
  IFS=',' read -r -a parts <<< "$str"

  model="${parts[0]}"
  batch="${parts[1]}"

  docker run --ipc=host --ulimit memlock=-1 --gpus '"device=1"' cv-benchmark --model $model --batch-size $batch --n-iter $N_ITERS --precision $PRECISION
done
