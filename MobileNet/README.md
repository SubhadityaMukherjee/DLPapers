**[5]** Mobile Net
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (2017), Andrew G. Howard et al.
[Paper](https://arxiv.org/pdf/1704.04861.pdf)


## Notes
- All layers except first(normal one for that) :nxm Depthwise Conv->BN->ReLU->1x1 Conv->BN->ReLU
- Depthwise separable filters : Add group = no of input channels in Conv2d
  - Splits it into layers
    - filtering layer
    - combining layer
  - use groups when going from lower channels to higher
  - Depthwise convolutions + pointwise convolutions
- use 1x1 convs  
- Features:
  - Smaller net
  - More Efficient convs
  - Model size
- AvgPool at the end
-RMSProp
- Very little or no weight decay for Depthwise filters
> In effect-> Width, resolution multiplier
