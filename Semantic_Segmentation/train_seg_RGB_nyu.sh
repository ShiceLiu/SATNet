#!/bin/bash


NYUIMAGE_PATH=/home/jason/NYUv2/NYU_images
CUDA_VISIBLE_DEVICES=0 python seg_RGB_nyu.py $NYUIMAGE_PATH 2>&1 | tee logs/seg_RGB_nyu.log