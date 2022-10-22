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

import matplotlib.pyplot as plt
from PIL import Image

modelCNN = CNN()
print(modelCNN)
path = './ILSVRC2012_val_00006543.JPEG'
img = Image.open(path)
toTensor = transforms.ToTensor()
trans = transforms.CenterCrop(32)
img = toTensor(img)
img = trans(img)
img=img.unsqueeze(0)

y=modelCNN(img)
print(y)