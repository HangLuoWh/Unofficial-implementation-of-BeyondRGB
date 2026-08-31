# Unofficial Implementation of BeyondRGB 
This is an unofficial implementation of the method described in the paper: Beyond RGB: A Real World Dataset for Multispectral Imaging in Mobile Devices.

在该论文中，作者采集了一个多光谱数据集，并提出了一种利用多光谱图像来估计光源SPD的方法。论文并未公开方法的源代码，因此我们复现了此方法。在BeyondRGB数据集测试复现的结果，所得到的结果如下：

**Lab test**
Mean: 5.8065° Std: 3.5719° Median: 5.1756° Trimean: 5.1198° Best25: 2.6419° Worst25: 10.3957 ° 95%: 12.1240°

**Field test** 
Mean: 6.7013° Std: 3.0769° Median: 6.2396° Trimean: 6.2598° Best25: 3.3886° Worst25: 10.6575 ° 95%: 13.5490°

此结果和原文汇报的结果较为接近。

# Requirement
1. torch
2. torchvision
3. numpy
4. opencv-python
5. h5py
6. PyYAML
7. tqdm
8. tensorboard

# Implementaion platform
- Operation System: Ubuntu 24.04.4 
- CPU: Intel® Core™ i5-8500 × 6
- GPU: NVIDIA GeForce RTX™ 2070 SUPER 8G

