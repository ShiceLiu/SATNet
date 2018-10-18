#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python SATNet_ThinkNetFuse.py 2>&1 | tee logs/SATNet_ThinkNetFuse.log