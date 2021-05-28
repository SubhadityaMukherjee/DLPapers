
**[2]** VGG net
 Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
 [Paper](https://arxiv.org/pdf/1409.1556.pdf)

## Notes
### Part1
- max pool: 2, 2
- conv: stri = 1, pad =1 ,ks = 3
- Always add ReLU
- Not all layers have max pool
- Adding 1x1 layers increases non linearity
> Network in network - 1x1
- mom = .9, decay = 5*10^-4, dropout = 0.5
- learning rate decay was used
- Improve convergence
  - Implicit regularization -> greater depth, smaller filter
  - Pre init of some layers
- Adding small amounts of noise to image -> increases accuracy
- Fusion
  - Averaging best soft max parts of multiple performing models

### Part2
- Object localization
- Euclidean loss -> penalizes deviation of bounding box
- No scale jittering
