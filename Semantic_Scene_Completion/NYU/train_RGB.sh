#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python SATNet_RGB.py 2>&1 | tee logs/SATNet_RGB.log