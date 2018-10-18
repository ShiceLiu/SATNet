#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python SATNet_Depth.py 2>&1 | tee logs/SATNet_Depth.log