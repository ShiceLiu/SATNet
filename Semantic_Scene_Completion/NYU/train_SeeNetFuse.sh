#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python SATNet_SeeNetFuse.py 2>&1 | tee logs/SATNet_SeeNetFuse.log