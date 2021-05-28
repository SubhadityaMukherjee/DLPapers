**[27]** HRNet (WIP)
- Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., ... & Liu, W. (2020). Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence. [Paper](https://arxiv.org/pdf/1908.07919)

# Notes

-  first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions in series and then recover the high-resolution representation from the encoded low-resolution representation.
- maintains high-resolution representations through the whole process.
-  Connect the high-to-low resolution convolution streams in parallel; (ii) Repeatedly exchange the information across resolutions
-  semantically richer and spatially more precise.
-  human pose estimation, semantic segmentation, and object detection
-  HRNetV1, only outputs the high-resolution representation computed from the high-resolution convolution stream. We apply it to human pose estimation
-  HRNetV2, combines the representations from all the high-to-low resolution parallel streams. We apply it to semantic segmentation
-  stem, which consists of two stride 2 3x3 convolutions decreasing the resolution to 1/4 , and subsequently the main body that outputs the representation with the same resolution ( 1/4 ).
-  We start from a high-resolution convolution stream as the first stage, gradually add high-to-low resolution streams
-   (a) HRNetV1: only output the representation from the high-resolution convolution stream. (b) HRNetV2: Concatenate representations that are from all 1x1 convolution is not shown for clarity). (c) HRNetV2p: form a feature pyramid from the representation by HRNetV representations
-   Heads -> HRNetV1. The output is the representation only from the high-resolution stream. Other three representations are ignored. HRNetV2. We rescale the low-resolution representations through bilinear upsampling without changing the number of channels to the high resolution, and concatenate the four representations, followed by a 1x1 convolution to mix the
four representations. HRNetV2p. We construct multi-level representations by downsampling the high-resolution representation output from HRNetV2 to multiple levels. .
-    parallel convolution of the modularized block
-    multi-resolution parallel convolutions, and multi-resolution fusion  The multi-resolution parallel convolution resembles the group convolution. It divides the input channels into several subsets of channels and performs a regular convolution over each subset over different spatial resolutions separately, while in the group convolution, the resolutions are the same
-    multi-resolution fusion unit resembles the multibranch full-connection form of the regular convolution
- multi-resolution fusion needs to handle the resolution change.
- Pose estimation -> transform this problem to estimating K heatmaps where each heatmap Hk indicates the location confidence of
the k th keypoint.
- Semantic seg -> Semantic segmentation is a problem of assigning a class label to each pixel.  feed the input image to the HRNetV2 and then pass the resulting 15C -dimensional representation at each position to a linear classifier with the softmax loss to predict the segmentation maps. The segmentation maps are upsampled (4 times) to the input size by bilinear upsampling for both training and testing
- We pretrain our network, which is augmented by a classification head shown in Figure 11, on ImageNet/ The
classification head is described as below. First, the four-resolution feature maps are fed into a bottleneck and the output
channels are increased from C , 2C , 4C , and 8C to 128, 256, 512, and 1024, respectively. Then, we downsample the high-
resolution representation by a 2-strided 3x3 convolution outputting 256 channels and add it to the representation of the
second-high-resolution. This process is repeated two times to get 1024 feature channels over the small resolution. Last, we
transform the 1024 channels to 2048 channels through a 1x1 convolution, followed by a global average pooling operation.
The output 2048-dimensional representation is fed into the classifier.

- For semantic segmentation, the time cost of the HRNettraining is slightly smaller and for inference significantly smaller than PSPNet and DeepLabv3
- For object detection, the time cost of the HRNet for training is larger than ResNet based networks and smaller than ResNext based
networks, and for inference the HRNet is smaller for similar GFLOPs
- For human pose estimation, the time cost of the HRNet for training is similar and for inference larger; and the time cost of the HRNet for training and inference in the MXNet platform is similar as SimpleBaseline
