#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset clevr-128-object-att --net res101 \
                   --bs 1 --nw 0 \
                   --lr 0.001 --lr_decay_step 5 \
                   --cuda
