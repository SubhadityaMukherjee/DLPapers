**[17]** Semantic segmentation DeepLab
- Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence, 40(4), 834-848. [Paper](https://arxiv.org/pdf/1606.00915)

# Notes
- Convolution with upsampled filters, or ‘atrous convolution’, as a powerful tool in dense prediction tasks
- Atrous convolution allows us to explicitly control the resolution at which feature responses are computed
- Effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation.
- Atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales.
- ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales.
- Combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance.
-  However, the repeated com- bination of max-pooling and striding at consecutive layers of these networks reduces significantly the spatial resolution of the resulting feature maps, typically by a factor of 32 across each direction in recent DCNNs.
- Atrous  allows us to compute the responses of any layer at any desirable resolution. It can be applied post-hoc, once a network has been trained, but can also be seamlessly integrated with training.
- For example, in order to double the spatial density of computed feature responses in the VGG-16 or ResNet-101 networks, we find the last pooling or convolutional layer that decreases resolution (’pool5’ or ’conv5 1’ respectively), set its stride to 1 to avoid signal decimation, and replace all subsequent convolutional layers with atrous convolutional layers having rate r = 2
- Using atrous convolution to increase by a factor of 4 the density of computed feature maps, followed by fast bilinear interpolation by an additional factor of 8 to recover feature maps at the original image resolution
- R-CNN spatial pyramid pooling method showed that regions of an arbitrary scale can be accurately and efficiently classified by resampling convolutional fea- tures extracted at a single scale
- Using variant of their scheme which uses multiple parallel atrous convolutional layers with different sampling rates. The fea- tures extracted for each sampling rate are further processed in separate branches and fused to generate the final result.
- To overcome these limitations of short-range CRFs, we integrate into our system the fully connected CRF model
