**[36]** LightSeg (only notes for now)
- Emara, T., Abd El Munim, H. E., & Abbas, H. M. (2019, December). LiteSeg: A Novel Lightweight ConvNet for Semantic Segmentation. In 2019 Digital Image Computing: Techniques and Applications (DICTA) (pp. 1-7). IEEE. [Link](https://arxiv.org/pdf/1912.06683) 

(P.S. This was done for an internship so is a bit more formal than usual)

# Notes

TL;DR -> Faster Semantic Segmentation with a modified ASPP module(from the [DeepLab](https://arxiv.org/pdf/1606.00915) paper) + MobileNetV2

## Objective
- Efficiency (To run on edge devices)
- Better performance (aka accuracy)
- Assign a label to every pixel
- Real time

## Modules used

### Atrous -> Dilated conv 
    - Decrease receptive field
    - Upsample by adding 0s b/w two filter vals along spatial dimensions with a dilation factor
    - Note for Pytorch -> Conv2d has dilation as a param
### ASPP
    - Take feature map -> Add 4 parallel atrous convs with different rates
### Deeper Atrous Spatial Pooling(DASPP)
    - Take ASPP
    - Add 3x3 convs after 3x3 atrous convs
### Depthwise separable conv
    - Replace normal convs
    - Split input and output into channels
    - Convolve pointwise
    - Note for Pytorch -> Add groups = no of in_channels in Conv2d
### Residuals
    - Take residual/skip connections 
      - H(X) = F(X) + X  ; H(X) is output; F(X) is residual  ; X is input feature map
    - Long -> Across larger no of layers
    - Short -> Smaller no of layers (as memory units)
    - Fuse them both 
    - Use 1x1 convs
    - End up with richer features
### Encoder
    - MobileNetV2 with output stride of 32
### Decoder
    - DeepLabv3+ decoder with ASPP modified to DASPP

## Whats new
- "Deeper" ASPP module
- Use DeepLabv3+/Mobile net etc as backbone for decoder
- Use long + short residuals (From resnet paper)
- Use Depthwise separable
- Modify strides for efficiency

## Misc training info
- SGD + Nesterov (0.9)
- Something similar to cyclic lr starting from 10e-7 
- Use weight decay of 4*10e-5
- Metric for Predicted bounding box vs ground truth : mIOU -> Mean of (Area of overlap/ Area of union)

## Results (mIOU)
- Cityscapes using MobileNet enc -> 66.48%
- 161fps for 360x640
