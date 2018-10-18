#!/bin/bash


NYUIMAGE_PATH=/home/jason/NYUv2/NYU_images
CUDA_VISIBLE_DEVICES=0 python seg_depth_nyu.py $NYUIMAGE_PATH 2>&1 | tee logs/seg_depth_nyu.log