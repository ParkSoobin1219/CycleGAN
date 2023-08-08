#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
 --dataroot ./horse2zebra \
 --name horse2zebra_cycleGAN \
 --model cycle_gan \
 --no_dropout \
 --lambda_A 2.0 \
 --lambda_B 2.0 
