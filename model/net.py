import torch
import torch.nn as nn
from torchvision import models
import constants as c

# Feature Extractor
class FeatExt(nn.Module):
  def __init__(self, feat_model):
    super().__init__()
    self.conv = nn.Sequential(*list(feat_model.children())[:-1])
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.relu(x)
    return x.view(x.shape[0], -1)

# Classifier Model
class ClfModel(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size, 1024)
    self.fc2 = nn.Linear(1024, c.num_class)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
