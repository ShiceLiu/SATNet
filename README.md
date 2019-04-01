# See and Think: Disentangling Semantic Scene Completion

This repository contains training and testing code for our paper on semantic scene completion, which leverages a disentangled framework to produce more accurate completions. More information about the project can be found in [our paper](https://papers.nips.cc/paper/7310-see-and-think-disentangling-semantic-scene-completion.pdf) and the 3-minute [video](https://youtu.be/YXvniY2U5ml).

![framework](image/framework.png)

If you find SATNet useful in your research, please cite:

	@inproceedings{liu2018see,
	  title={See and Think: Disentangling Semantic Scene Completion},
	  author={Liu, Shice and Hu, Yu and Zeng, Yiming and Tang, Qiankun and Jin, Beibei and Han, Yinhe and Li, Xiaowei},
	  booktitle={Advances in Neural Information Processing Systems},
	  pages={261--272},
	  year={2018}
	}

### Requirement

Operating System: Ubuntu 14.04 LTS (or higher)

Deep Learning Framework: PyTorch 0.4.0a0+408c84d (Some data structures and functions might be different in other versions)

Python: 2.7 (Some functions might be different in Python 3.*)

Python Package: Numpy, OpenCV, Visdom, torchvision

Memory: 16GB

GPU: 8GB Memory for single-branch structure and 11GB~14GB Memory for double-branch structure

### Introduction

This project mainly consists of two parts, semantic segmentation and semantic scene completion. The semantic segmentation results will accelerate the convergence speed of semantic scene completion.

If you want to train either part, you should download the [datasets](#Datasets) and fix the file path in the 'config.py' and the certain '.sh'.

If you want to test either part only, you should download the [pretrained models](#Pretrained-Models).

### Datasets

The datasets, used in the project, are stored in the baiduyun. The URL is [https://pan.baidu.com/s/1MEZ-HY4La7EwlS0I7lpOxw](https://pan.baidu.com/s/1MEZ-HY4La7EwlS0I7lpOxw).

For convenience, we split the big file into several smaller files by the instruction 'split' in the Ubuntu. Therefore, before using the big file, we need to merge these small files by the instruction 'cat' in the Ubuntu.

For example, we need to run the command 'cat myselect_suncg.zip.* > myselect_suncg.zip' to merge them.

Besides, for semantic segmentation on the SUNCG-RGBD dataset, RGB-D images and their labels are stored in the baiduyun too. The URL is [https://pan.baidu.com/s/1vUt09GlkC1lPFRm8zXofZA](https://pan.baidu.com/s/1vUt09GlkC1lPFRm8zXofZA). You should unzip the 'SUNCGRGBD_images.zip' file and make a new directory named '/home/jason/lsc_datasets/' (This directory name is consistent with '~/SATNet_datasets/image_list_val.txt' and so on.). Then put all the files in the 'SUNCGRGBD_images' directory into this new directory. 

### Pretrained Models

The pretrained models are also stored in the baiduyun. The URL is [https://pan.baidu.com/s/1wk4-ShGW2PNUL3eliNa1Hg](https://pan.baidu.com/s/1wk4-ShGW2PNUL3eliNa1Hg).

### Data Preprocessing

We have uploaded the data-preprocessing code in the 'util' directory. The data-preprocessing procedure is to convert '.bin' to '.npz'. There are several approaches to achieve it and we share one of them. We write a CUDA+C library and use Python to generate '.npz'. Unfortunately, users have to read the code and modify the certain parts to make the code run correctly.

### Organization

	Semantic_Segmentation : 
	  seg_* : the front-end of the certain task.
	  engine* : the back-end of the certain task.
	  *.sh : the interface of training.

	Semantic_Scene_Completion :
	  NYU :
	    seg_* : the front-end of the certain task.
	    engine* : the back-end of the certain task.
	    *.sh : the interface of training.
	    gen_result_* : to generate the results of the certain task.
	    pretrained_models/ : training on the NYU is based on the pretrained models on the SUNCG.

	  SUNCG_D :
	    seg_* : the front-end of the certain task.
	    engine* : the back-end of the certain task.
	    *.sh : the interface of training.
	    gen_result_* : to generate the results of the certain task.

	  SUNCG_RGBD :
	    seg_* : the front-end of the certain task.
	    engine* : the back-end of the certain task.
	    *.sh : the interface of training.
	    gen_result_* : to generate the results of the certain task.
	    eval_results.py : to evaluate the results with the groundtruth.
	    labels/ : the groundtruth.

### License

Code is released under the MIT License (refer to the LICENSE file for details)
