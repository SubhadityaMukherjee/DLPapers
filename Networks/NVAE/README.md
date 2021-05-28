**[29]** NVAE (WIP)
- Vahdat, A., & Kautz, J. (2020). NVAE: A Deep Hierarchical Variational Autoencoder. arXiv preprint arXiv:2007.03898. [Paper](https://arxiv.org/abs/2007.03898)

## Notes

- VAEs have the advantage of fast and tractable sampling and easy-to-access encoding networks
- carefully designing neural ar- chitectures for hierarchical VAEs
- Nouveau VAE
- depth-wise separable convolu- tions and batch normalization
- residual parameterization of Normal distributions and its training is stabilized by spectral regularization.
- first successful VAE applied to natural images as large as 256x256 pixels.
-  VAEs maximize the mutual information between the input and latent variables requiring the networks to retain the information content of the input data as much as possible.
-  VAEs often respond differently to the over-parameterization in neural networks.
-  The current state-of-the-art VAEs omit batch normalization (BN) to combat the sources of randomness that could potentially amplify their instability.
-  turns out BN is an important component of the success of deep VAEs. 
-  spectral regularization is key to stabilizing VAE training.
-   the case of VAEs with unconditional decoder, such long-range correlations are encoded in the latent space and are projected back to the pixel space by the decoder.
-  Our generative model starts from a small spatially arranged latent variables as z1 and samples from the hierarchy group-by-group while gradually doubling the spatial dimensions
-  capture global long-range correlations at the top of the hierarchy and local fine-grained dependencies at the lower groups.
- increas- ing the receptive field of the networks
- increasing the kernel sizes in the convolutional pat
-  depthwise convolutions outperform regular convolutions while keeping the number of parameters and the computational complexity orders of magnitudes smaller
-  BN has a negative during evaluation but not training
-  modify the momentum parameter of BN such that running statistics can catch up faster with the batch statistics. We also apply a regularization on the norm of scaling parameters in BN layers to ensure that a small mismatch in statistics is not amplified by BN
-  We also observe that the combination of BN and Swish outperforms WN and ELU activation
-  Similar to mobile net 2 with two additional BN layers at the beginning and the end of the cell and it uses Swish activation function and SE.
-  mixed precision
-  to fuse BN and Swish and we store only one feature map for the backward pass, instead of two
-  To bound KL, we need to ensure that the encoder output does not change dramatically as its input changes. This notion of smoothness is characterized by the Lipschitz constant.
-  apply a few additional normalizing flows to the samples generated at each group in q   
