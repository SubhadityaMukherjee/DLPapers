**[8]** Spatial Transformer Networks
- Jaderberg, M., Simonyan, K., & Zisserman, A. (2015). Spatial transformer networks. In Advances in neural information processing systems (pp. 2017-2025).
[Paper](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)

# Tricks
- Spatial transformer module
  - separate object pose and deformation (texture+shape)
  - try to restore original from warped
  - can be used in any existing network
  - Three parts:
    - Localization network
      - Input feature map -> hidden -> parameters of spatial transform
    - Grid generator
      - Set of points where image should be sampled for transform
    - Sampler
      - Output map is produced after combining both the outputs
