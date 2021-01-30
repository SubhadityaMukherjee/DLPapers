**[16]** Perceptual Loss
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016, October). Perceptual losses for real-time style transfer and super-resolution. In European conference on computer vision (pp. 694-711). Springer, Cham. [Paper](https://arxiv.org/pdf/1603.08155.pdf%7C)

# Notes
- network gives similar qual-
itative results but is three orders of magnitude faster
-  image trans-
formation network + loss network
- Image transformation network is a deep residual convolutional neural network
- The image transformation network is trained using stochastic gradient descent to minimize a weighted combination of loss functions
- Convolutional neural networks pretrained for image classification have already learned to encode the perceptual and semantic information we would like to measure in our loss functions.
- Image transformation
  - We do not use any pooling layers instead using strided and fractionally strided convolutions for in-network downsampling and upsampling.
  - All non-residual convolutional layers are followed by spatial batch normalization + ReLU except last layer -> tanh
-  For super-resolution with an upsampling factor of f , we use several residual blocks followed by log2 f convolutional layers with stride 1/2.
- For style transfer our networks use two stride-2 convolutions to downsample the input followed by several residual blocks and then two convolutional layers with stride 1/2 to upsample
- feature reconstruction loss is the (squared, normalized) Euclidean distance between feature representations
- style reconstruction loss is then the squared Frobenius norm of the dif- ference between the Gram matrices of the output and target images
-  The pixel loss is the (normalized) Euclidean distance between the output image ŷ and the target y
- Total Variation Regularization. To encourage spatial smoothness in the
output image
- we train super-resolution networks not with the per-pixel loss typically used [1] but instead with a feature reconstruction loss
- The feature reconstruction loss gives rise to a slight cross-hatch pattern visible under magnification
-  The pixel loss gives fewer visual artifacts and higher PSNR values but the feature loss does a better job at reconstructing fine details, leading to pleasing visual results
