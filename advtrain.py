import os
import sys

import torch
import torch.nn
import torch.optim as optim
import numpy as np

import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from targetmodels import CNN

modelCNN = CNN()
print(modelCNN)