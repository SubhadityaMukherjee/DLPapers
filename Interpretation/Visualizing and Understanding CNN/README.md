**[16]** Perceptual Loss (For super resolution)
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016, October). Perceptual losses for real-time style transfer and super-resolution. In European conference on computer vision (pp. 694-711). Springer, Cham. [Paper](https://arxiv.org/pdf/1603.08155.pdf%7C)


# Notes
- We train these models using a large set of N labeled
images {x, y}, where label yi is a discrete variable
indicating the true class. A cross-entropy loss func-
tion, suitable for image classification,
- map these activities back to the input pixel space, showing what input pattern originally caused a given activation in the feature maps
- A deconvnet can be thought of as a convnet model that uses the same components (filtering, pooling) but in reverse, so instead of mapping pixels to features does the opposite.
- Then we successively
(i) unpool, (ii) rectify and (iii) filter to reconstruct
the activity in the layer beneath that gave rise to the
chosen activation.
- In the convnet, the max pooling operation is non-invertible, however we can obtain an approximate inverse by recording the locations of the maxima within each pooling region in a set of switch variables.
- We pass the reconstructed signal through a relu non-linearity.
- To combat this, we renormalize each filter in the convolutional layers whose RMS value exceeds a fixed radius of 10−1 to this fixed radius.
-  Sudden jumps in appearance result from a change in the image from which the strongest activation originates. The lower layers of the model can be seen to converge within a few epochs.
- However, the upper layers only develop develop after a considerable number of epochs (40-50), demonstrating the need to let the models train until fully converged
- Small transformations have a dramatic effect in the first layer of the model, but a lesser impact at the top feature layer, being quasilinear for translation & scaling.
- The network output is stable to translations and scalings. In general, the output is not invariant to rotation, except for object with rotational symmetry
