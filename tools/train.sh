#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/Pancreas_CT.yaml


python -m torch.distributed.launch \
   --nproc_per_node=$1 \
   --master_addr=localhost \
   --master_port=$2 \
   main3D.py \
   --config=$config



















