#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_net.py \
                   --dataset colormnist-object-attr --net res101 \
                   --checksession 1 --checkepoch 19 --checkpoint 4999 \
                   --cuda
