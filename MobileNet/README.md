**[5]** Mobile Net
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (2017), Andrew G. Howard et al.
[Paper](https://arxiv.org/pdf/1704.04861.pdf)


## Tricks
- All layers except first(normal one for that) :nxm Depthwise Conv->BN->ReLU->1x1 Conv->BN->ReLU
- Depthwise : Add group = no of input channels in Conv2d
  - Splits it into parts
  - use groups when going from lower channels to higher
- use 1x1 convs  
