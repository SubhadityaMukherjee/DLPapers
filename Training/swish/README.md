**[24]** swish
- Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. arXiv preprint arXiv:1710.05941. [Paper](https://arxiv.org/pdf/1710.05941;%20http://arxiv.org/abs/1710.05941)

# Notes
- Supposedly does better than selu and relu sometimes (Nope)
- Slower than relu
- f (x) = x · sigmoid(βx)
- found using RL techniques using "core units"
-  simple search space inspired by the optimizer search space
of Bello et al. (2017) that composes unary and binary functions to construct the activation function
-  activation function is constructed by repeatedly composing the the “core
unit”, which is defined as b(u1 (x1 ), u2 (x2 )). The core unit takes in two scalar inputs, passes each
input independently through an unary function, and combines the two unary outputs with a binary
function that outputs a scalar
## Tips for activation fns
-  Complicated activation functions consistently underperform simpler activation functions,
potentially due to an increased difficulty in optimization. 
-  A common structure shared by the top activation functions is the use of the raw preactiva-
tion x as input to the final binary function
-  The searches discovered activation functions that utilize periodic functions, such as sin and
cos.
-  Functions that use division tend to perform poorly because the output explodes when the
denominator is near 0