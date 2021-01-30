**[21]** Unets
- Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. [Paper](https://arxiv.org/pdf/1505.04597.pdf)

# Notes

- ISBI cell tracking challenge 2015 
- localization, i.e., a class label is supposed to be assigned to each pixel.
- works with very few training images and yields more precise segmentations
- supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators. Hence, these layers increase the resolution of the output. In order to localize, high resolution features from the contracting path are combined with the upsampled output. A successive convolution layer can then learn to assemble a more precise output based on this information.
-  large number of feature channels, which allow the network to propagate context information to higher resolution layers.
- To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image.
- Use elastic deformations -> learn invariance
-  even x- and y-size.
-  pixel-wise soft-max over the final feature map combined with the cross entropy loss function

