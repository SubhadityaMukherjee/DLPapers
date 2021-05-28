**[31]** Computational Limits (Just notes)
- Thompson, N. C., Greenewald, K., Lee, K., & Manso, G. F. (2020). The Computational Limits of Deep Learning. arXiv preprint arXiv:2007.05558. [Paper](https://arxiv.org/pdf/2007.05558)

- Deep learning is quickly becoming unsustainable economically, environmentally and technically
- deep learning might soon become computationally constrained even though substantial improvements might be possible

# theory
- over parameterizing a neural network basically means that it would be given more parameters and there are data points
- a cost of training the neural network scales with the product of the number of parameters with the number of data points
- theoretically, It grows by at least the square of the number of data points in an over parameterized setting
- we should always be aware of a performance plateau
- as the amount of data increases, standard flexible models are performed expert models because they do not capture all the contributing factors
- traditional machine learning techniques to better when data is small and deep learning does better when there's a huge amount of data. This is because of over parameterization which makes use of implicit regularization 

# performance

- We find highly-statistically significant slopes and strong explanatory power (R2 between 29% and 68%) for all benchmarks except machine translation, English to German, where we have very little variation in the computing power used. 
- Object detection, named-entity recognition and machine translation show large increases in hardware burden with relatively small improvements
- polynomial models best explain this data, but that models implying an exponential increase in computing power as the right functional form are also plausible.
- more-optimistic model, it is estimated to take an additional 10^5× more computing to get to an error rate of 5% for ImageNet.
- fundamental rearchitecting is needed to lower the computational intensity
- For deep learning, these included mostly GPU and TPU implementations, although it has increasingly also included FPGA and other ASICs.
- analog hardware with in-memory computation, neuromorphic computing, optical computing , and quantum computing based approaches [90], as well as hybrid approaches
- quantum computing is the approach with perhaps the most long-term upside
- pruning” away weights ,quantizing the network, or using low-rank compression are important 
- overhead of doing meta learning or neural architecture search is itself computationally intense
- move to other, perhaps as yet undiscovered or underappreciated types of machine learning.
-  era when improvements in hardware perfor- mance are slowing. 