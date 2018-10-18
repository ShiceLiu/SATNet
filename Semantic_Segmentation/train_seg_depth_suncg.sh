#!/bin/bash

# Note that 'HHA_path' in the 'seg_depth_suncg.py' should be replaced to your local path.
SELECTEDIMAGE_PATH=/home/jason/lscCcode/selectImageFromRawSUNCG/selectedImage
CUDA_VISIBLE_DEVICES=0 python seg_depth_suncg.py $SELECTEDIMAGE_PATH 2>&1 | tee ./logs/seg_depth_suncg.log