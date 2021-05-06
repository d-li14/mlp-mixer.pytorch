# PyTorch implementation of MLP-Mixer

[MLP-Mixer: an all-MLP architecture](https://arxiv.org/abs/2105.01601) composed of alternate token-mixing and channel-mixing operations.

* The `token-mixing` is like [involution](https://arxiv.org/abs/2103.06255) in terms of channel-agnostic weights, but involution is more flexible with spatial-specific weights. This difference makes involution more friendly to transfer to downstream tasks, such as detection and segmentation.

* The `channel-mixing` is like 1x1 convolution, permiting channel information exchange.

The combination of the above two is similar to replacing 3x3 convolution in the ResNet bottleneck block with involution, while maintaining the 1x1 convolution, giving rise to our convolution-free, attention-free architecture [RedNet](https://github.com/d-li14/involution).

Anyway, the take-home message is common: **fully-MLP based architecture could rival convolution or self-attention based architectures**.

## Ackowlegement
The implementation is based on the JAX/Flax code in the Appendix of the original paper.
