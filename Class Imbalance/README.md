**[15]** Class Imbalance Problem
- Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249-259. [Paper](https://arxiv.org/pdf/1710.053

## Notes
-  the effect of class imbalance on classification performance is detrimental
- the method of addressing class imbalance that emerged as dominant in almost all analyzed scenarios was oversampling
- oversampling does not cause overfitting of CNN
- thresholding should be applied to compensate for prior class probabilities when overall number of properly classified cases is of interest.

- oversampling
  -  random minority oversampling, which simply replicates randomly selected samples from minority classes.
  - Cluster-based oversampling first clusters the dataset and then oversamples each cluster separately
  -  An oversampling approach specific to neural networks optimized with stochastic gradient descent is class aware sampling. The main idea is to ensure uniform class distribution of each mini-batch and control the selection of examples from each class.
- Thresholding
  - It is applied in the test phase and involves changing the output class probabilities. There are many ways in which the network outputs can be adjusted
- Cost sensitive learning.
  - This method assigns different cost to misclassification of examples from different classes
- One-class classification.
  - In the context of neural networks it is usually called novelty detection. This is a concept learning technique that recognizes positive instances rather than discriminating between two classes. Autoencoders used for this purpose are trained to perform autoassociative mapping, i.e. identity function
- effect of imbalance is significantly stronger for the task with higher complexity
- Undersampling showed a generally poor performance.
- not only the total number of examples matters but also its distribution between classes.
- thresholding worked particularly well when applied jointly with oversampling.
