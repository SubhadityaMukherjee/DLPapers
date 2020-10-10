**[35]** Federated Learning (original paper)
- Konečný, J., McMahan, H. B., Yu, F. X., Richtárik, P., Suresh, A. T., & Bacon, D. (2016). Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492.  [Paper](https://arxiv.org/pdf/1610.05492)

I also wrote a detailed blog post for this [Post](https://medium.com/datadriveninvestor/federated-learning-d466aba3abbd?source=friends_link&sk=91072a0b0d19c2f6641af5c471159c80)

# Notes from the paper

- training data remains distributed over a large number of clients each with unreliable and relatively slow network connections
- computes an update to the current model based on its local data, and communicates this update to a central server, where the client-side updates are aggregated to compute a new global model. 
- slight degradation in convergence speed 

## Types of updates

- Structured updates, where we directly learn an update from a restricted space that can be
parametrized using a smaller number of variables.
- Sketched updates, where we learn a full model update, then compress it before sending to
the server.

## Structured Update

- we train directly the updates of this structure
- Random mask. We restrict the update Hit to be a sparse matrix, following a pre-defined random
sparsity pattern 

##  Sketched update

-  first computes the full Hit during local training without any constraints, and then approximates, or encodes, the update in a (lossy) compressed form before sending to the server. The server decodes the updates before doing the aggregation. 
- Subsampling. Instead of sending Hit , each client only communicates matrix Ĥit which is formed
from a random subset of the (scaled) values of Hit.
- Quantize the weights
- Improving the quantization by structured random rotations. The above 1-bit and multi-bit quan-
tization approach work best when the scales are approximately equal across different dimensions.
- In the decoding phase, the server needs to perform the inverse rotation before aggregating all the
updates. 

