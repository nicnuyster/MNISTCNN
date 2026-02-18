import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import cv2 as cv

def ImportsPrints():

    print(f"Torch - {torch.version}")
    print(f"CV2   - {cv.__version__}")
    print(f"matplotlib - {plt.__name__}")

class MnistCNN():
    
    Batches = 64
    Learn_rate = 0.001
    Epoch_num = 5
    Class_num = 10
    Dropout_r = 0.5

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, NBatches = 64):
        self.Batches = NBatches
        self.transforms

    def PreParamsPrint(self):

        print 
        print(f"device - {self.Device}")

        

if __name__ == '__main__':
    MCNN = MnistCNN()
    MCNN.Mtest()
    MCNN.ModelParramPrint()