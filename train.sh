#!/usr/bin/env bash

CONFIG=$1

#python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321  basicsr/train.py -opt $CONFIG --launcher pytorch
python --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt Dehazing/Options/MB-TaylorFormer-B.yml --launcher pytorch

#CUDA_VISIBLE_DEVICES=0 python -Xfaulthandler basicsr/train.py -opt Dehazing/Options/MB-TaylorFormerV2-B.yml > train.log 2>&1 &
