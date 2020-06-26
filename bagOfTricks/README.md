**[24]** Bag Of Tricks
- He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019). Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 558-567). [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019    _paper.pdf)

## Training procedure
- 1. Randomly sample an image and decode it into 32-bit
floating point raw pixel values in [0, 255].
- 2. Randomly crop a rectangular region whose aspect ratio
is randomly sampled in [3/4, 4/3] and area randomly
sampled in [8%, 100%], then resize the cropped region
into a 224-by-224 square image.
- 3. Flip horizontally with 0.5 probability.
- 4. Scale hue, saturation, and brightness with coefficients
uniformly drawn from [0.6, 1.4].
- 5. Add PCA noise with a coefficient sampled from a nor-
mal distribution N (0, 0.1).
- 6. Normalize RGB channels by subtracting 123.68,
116.779, 103.939 and dividing by 58.393, 57.12,
57.375, respectively.
- initialized with the Xavier algorithm
- All biases are initialized to 0. For batch normaliza-
tion layers, γ vectors are initialized to 1 and β vectors0.
- Nesterov Accelerated Gradient (NAG) descent 
- The learning rate is initialized to 0.1 and divided by 10 at the
30th, 60th, and 90th epochs

## Large batch training
- For convex problems, convergence
rate decreases as batch size increases
-  In other
words, for the same number of epochs, training with a large
batch size results in a model with degraded validation accu-
racy compared to the ones trained with smaller batch sizes.
- In other words, a large batch
size reduces the noise in the gradient, so we may increase
the learning rate to make a larger progress along the op-
posite of the gradient direction
-  a gradual warmup
strategy that increases the learning rate from 0 to the initial
learning rate linearly.

## Zero gamma
-  In the zero γ initialization heuristic, we initialize
γ = 0 for all BN layers that sit at the end of a residual block.
Therefore, all residual blocks just return their inputs, mim-
ics network that has less number of layers and is easier to
train at the initial stage.

## No bias decay
-  Other parameters, including the biases
and γ and β in BN layers, are left unregularized

## FP16
- the overall training speed is acceler-
ated by 2 to 3 times after switching from FP32 to FP16 on
V100

## XResnet
- Empirically, we found
adding a 2×2 average pooling layer with a stride of 2 before
the convolution, whose stride is changed to 1, works well
in practice and impacts the computational cost little

## One cycle
- As can be seen, the cosine decay
decreases the learning rate slowly at the beginning, and then
becomes almost linear decreasing in the middle, and slows
down again at the end.

## Label Smoothing CE
- It is clear that with label smoothing the distribution centers
at the theoretical value and has fewer extreme values.

## Distillation
- we use a teacher model
to help train the current model, which is called the student
model. The teacher model is often a pre-trained model with
higher accuracy, so by imitation, the student model is able
to improve its own accuracy while keeping the model com-
plexity the same.
-  distillation loss to penalize
the difference between the softmax outputs from the teacher
model and the learner model.

## Use mixup


