**[10]** VAE (Auto-Encoding Variational Bayes)
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
[Paper](https://arxiv.org/pdf/1312.6114.pdf?source=post_page---------------------------)

# Tricks
- Since MSE -> blurry results
- Can compare image directly with output unlike GAN
- Loss
  - KL Divergence
    - latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)  
  - generation_loss = mean(square(generated_image - real_image))  
  - latent_loss = KL-Divergence(latent_variable, unit_gaussian)  
  - loss = generation_loss + latent_loss  
- Estimator : Stochastic Gradient Variable Bayes Estimator
- Reparameterize
- NN as probabilistic encoder


- Architecture
- ![arch](model.png)
