# GANs Implementation

In this repository, there is a series of implementation for diverse GAN architectures.

## Vanillia GAN

First implementation follows the algorithm presented in the [Generative Adversarial Nets paper](https://arxiv.org/pdf/1406.2661.pdf)

Some worth mentioning changes. Due to the often CUDA out of memory errors, this implementation uses Automatic Mixed Precision and Gradient Accumulation.
Warning! On an Tesla K80 GPU/T4 GPU(Colab free) and a batch size of 32, the training time is approximately 4,5h/3h (400 epochs).