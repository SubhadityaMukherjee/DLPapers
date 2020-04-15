**[13]** One cycle paper
- Smith, L. N. (2017, March). Cyclical learning rates for training neural networks. In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 464-472). IEEE.
[Paper](https://arxiv.org/pdf/1506.01186.pdf%EF%BC%89%EF%BC%8C%E8%BF%99%E7%A7%8D%E5%A5%87%E6%8A%80%E6%B7%AB%E5%B7%A7%E5%B0%86%E8%8E%B7%E5%BE%97%E6%9B%B4%E9%AB%98%E7%9A%84%E6%B5%8B%E8%AF%95%E5%87%86%E7%A1%AE%E7%8E%87%EF%BC%8C%E4%BD%86%E6%98%AF%E4%BD%A0%E7%9C%8B%E8%BF%99%E4%B8%AAlearning)

# Notes
- Practically eliminates the need to experimentally find the best values and
schedule for the global learning rates
- learning rate cyclically vary between reasonable boundary values.
- paper also describes a simple way to estimate “reasonable bounds” – linearly increasing the learning rate of the network for a few epochs.
- increasing the learning rate might have a short term negative effect and yet achieve a longer term beneficial effect.
- learning rate vary within a range of values rather than adopting a stepwise fixed or exponentially decreasing value
- triangular window
- Saddle points have small gradients that slow the learning process. However, increasing the learning rate allows more rapid traversal of saddle point plateaus.
- it is likely the optimum learning rate will be between the bounds and near optimal learning rates will be used throughout training.
- experiments show that it often is good to set stepsize equal to 2 − 10 times the number of iterations in an epoch
- a single LR range test provides both a good LR
value and a good range.
> Note : Dont forget to use a range of lr instead of just one 
