#!/bin/bash


NYUIMAGE_PATH=/home/jason/NYUv2/NYU_images
CUDA_VISIBLE_DEVICES=1 python seg_fuse_nyu.py $NYUIMAGE_PATH 2>&1 | tee logs/seg_fuse_nyu.log