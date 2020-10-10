**[34]** Google Keyboard Federated Learning(Notes only. refer to [35] for code)
- Chen, M., Mathews, R., Ouyang, T., & Beaufays, F. (2019). Federated learning of out-of-vocabulary words. arXiv preprint arXiv:1903.10635.  [Paper](https://arxiv.org/pdf/1903.10635)

# Paper notes

- character-level recurrent neural network is able to learn out- of-vocabulary
- High-frequency words can be sampled from the trained generative model by drawing from the joint posterior directly
- Studied using 1. Simulation, 2. Gboard App
- Learning frequently typed words
- Words missing from the vocabulary cannot be predicted on the keyboard suggestion strip
- neural machine translation (NMT), rely on a vocabulary to encode words during endto-end training 
- uploading only ephemeral model updates to the server for aggregation, and leaving the users’ raw data on their device. 
-  the privacy risk of unintended memorization still exists
- We further show that the top sampled words are very meaningful and are able to capture words we know to be trending in the news at the time of the experiments

## LSTM Modeling

- In this work we use a variant of LSTM with a Coupled Input and Forget Gate (CIFG) (Greff et al., 2017), peephole con- nections (Gers and Schmidhuber, 2000) and a pro- jection layer (Sak et al., 2014) 
- Cross entropy loss
- This parallel sampling approach avoids the dependency between each sampling thread, which might occur in beam search or shortest path search sampling
- Adaptive L2 -norm clipping is performed on each client’s gradient, as it is found to improve the robustness of model convergence
- The larger model (on training with Reddit data) does not lead to significant gains. Momentum and adaptive clipping lead to faster convergence and more stable performance.

# Conclusion

- We also perform live experiments with on-device data from 3 populations of Gboard users and demonstrate that this method can learn OOV words effectively in a real-world setting.
