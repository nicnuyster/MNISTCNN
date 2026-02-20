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

def ImportsPrint():

    print(f"Torch - {torch.version}")
    print(f"CV2   - {cv.__version__}")
    #print(f"PLT   - {plt.__version__}")
    print(f"PLT   - {plt.__file__}")
    print(f"NumPy - {np.__version__}")

class MnistCNN():
    
    Batches = 64
    Learn_rate = 0.001
    Epoch_num = 5
    Class_num = 10
    Dropout_r = 0.5

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # def layers
    def __init__(self, NBatches = 64):

        super(MnistCNN, self).__init__()

        self.Batches = NBatches
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # convs layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # 1x28x28 -> 16x14x14 -> 32x7x7
        
        self.feature_size = 32 * 7 * 7

        #FC
        self.fc1 = nn.Linear(self.feature_size, 64)
        self.fc2 = nn.Linear(64, self.Class_num)
        self.drop = nn.Dropout(self.Dropout_r)
        self.relu = nn.ReLU()

        #training

        self.model = model


    # MNIST
    def DataLoading(self):
        
        dataset_train = torchvision.datasets.MNIST(root = './data', train = True, transform = self.transform, download = True)
        dataset_test = torchvision.datasets.MNIST(root = './data', train = False, transform = self.transform, download = True)

        loader_train = DataLoader(dataset_train, batch_size = self.Batches, shuffle = True)
        loader_test = DataLoader(dataset_test, batch_size = self.Batches, shuffle = True)

        return loader_train, loader_test

    # Forward 
    def forward(self, x):
        
        # conv
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        #flat
        x = x.view(-1, self.feature_size)
        #fc
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x

    def PreParamsPrint(self):

        print(f"device - {self.Device}")

        

if __name__ == '__main__':
    ImportsPrint()

    MCNN = MnistCNN()
    MCNN.PreParamsPrint()