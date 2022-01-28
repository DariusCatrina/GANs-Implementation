# The models
# Generator - generates fake images
# G(Noise) - Fake Image
# Discriminator - distinguishs between fake & real images

import torch.nn as nn
import torch

class Generator(nn.Module):
  def __init__(self, noise_size, img_features):
    super().__init__()
    self.model = nn.Sequential(
          nn.Linear(noise_size, 256),
          nn.LeakyReLU(0.01), #it helps the gradients flow easier through the architecture
          nn.Linear(256, img_features),
          nn.Tanh(),  #for output [-1, 1]
      )

  def forward(self, noise):
    return self.model(noise)


class Discriminator(nn.Module):
  def __init__(self, img_features):
      super().__init__()
      self.model = nn.Sequential(
          nn.Linear(img_features, 128),
          nn.LeakyReLU(0.01), #helps the gradients flow easier through the architecture
          nn.Linear(128, 1),
          nn.Sigmoid()
      )

  def forward(self, img):
    img = img.view(-1, 28*28)
    return self.model(img).view(-1)
