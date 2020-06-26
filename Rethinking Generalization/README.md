**[28]** Understanding Deep learning requires rethinking generalization (Just notes)
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016). Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530. [Paper](https://arxiv.org/pdf/1611.03530.pdf?from=timeline&isappinstalled=0)

> This is in a different format as I had actually written this while applying somewhere 

## Motivation
Neural networks are essentially black boxes and many efforts to understand how the results which are obtained are influenced by the parameters in a network. This paper aims to identify the effects of regularization techniques on a neural network and prove that the usual understanding is flawed. The authors also try to prove the expressivity of a network independent of its regularization.

## Insights

The major question which is answered in this paper is that the effect of regularization techniques such as weight decay, augmentation, and dropout is not what It was assumed to be because the network tends to perform as well without them. The networks experimekief it was shown that neural networks can fundamentally fit a random collection of labels assigned to a map which shows its generalizability. Most of these experiments were carried out without any change of the hyperparameters or and by using stochastic gradient descent which is not a very advanced training algorithm. It was observed that the more noise that was added to the network led to a worsening of the generalization capabilities. This again proves the power of neural networks when it comes to being in a universal approximation function. The authors talk about the fact that explicit regularization is a tuning parameter and does not always lead to better performance on the validation set. This work also proves that a simple kernel actually performs really well without any of the mentioned regularizations for simple tasks.

Another question that was attempted to talk about in this paper was that of the model capacity of these networks. This basically means identifying the complexity of the network required to generalize to a specific task. The labels were partially changed or randomized along with the images themselves being distorted in a certain way. The network still generalized to these random changes which are very contradictory to the way that we generally view them. The randomness of the labels corrupted the mapping between the labels and the data but the network still generalized and converged which is shocking. On a simple classification data set with 10 classes, older architectures such as the Alex net and multilateral perceptrons all managed to converge. Another metric that was talked about was that of the sensitivity of the network to a change in the data for a single object. It was not possible to fundamentally prove how regularization affects this and further work must be done. The authors confirm that the technique of early stopping might have an impact on the performance while batch normalization improves the performance but is not needed specifically for generalization. The others concluded by proving the theorem of the existence of a double layer neural network just using rectified linear units and a 2n + d weight which could potentially represent any function of dimensions and size of n.

These networks still remain extremely hard to dig into and understand and efforts must be taken to identify the reasons for these networks doing what they do. Many things must be called into question which was long-standing facts and widely used.



