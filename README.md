# Unofficial Implementation of BeyondRGB 
This is an unofficial implementation of the method described in the paper: Beyond RGB: A Real World Dataset for Multispectral Imaging in Mobile Devices.

In this paper, the authors collected a multispectral dataset and proposed a method for estimating the spectral power distribution (SPD) of a light source from multispectral images. Since the source code for the method was not released, we reproduced it. The results obtained by evaluating our reproduction on the BeyondRGB dataset are shown below and are close to those reported in the original paper.

**Lab test**  
Mean: 5.8065°  
Std: 3.5719°  
Median: 5.1756°  
Trimean: 5.1198°  
Best25: 2.6419°  
Worst25: 10.3957°  
95%: 12.1240°  

**Field test**  
Mean: 6.7013°  
Std: 3.0769°  
Median: 6.2396°  
Trimean: 6.2598°  
Best25: 3.3886°  
Worst25: 10.6575°  
95%: 13.5490°  

# Requirements
1. torch
2. torchvision
3. numpy
4. opencv-python
5. h5py
6. PyYAML
7. tqdm
8. tensorboard

# Implementation Platform
- Operating System: Ubuntu 24.04.4 
- CPU: Intel® Core™ i5-8500 × 6
- GPU: NVIDIA GeForce RTX™ 2070 SUPER 8G

# How to Use the Code
First, download the BeyondRGB dataset and extract it into the current directory. To retrain a model, run `trainer.py`.

A pretrained model is provided in the `Result` directory. Run `test.py` to evaluate it directly.