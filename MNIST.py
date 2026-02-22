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

    Criteria = nn.CrossEntropyLoss()

    # hardware
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets
    dataset_train = torchvision.datasets.MNIST(root = './data', train = True, transform = Transform, download = True)
    dataset_test = torchvision.datasets.MNIST(root = './data', train = False, transform = Transform, download = True)

    # a slice of data
    loader_train = DataLoader(dataset = dataset_train, batch_size = Batches, shuffle = True)
    loader_test = DataLoader(dataset = dataset_train, batch_size = Batches, shuffle = True)

    # def layers
    def __init__(self):
        super(MnistCNN, self).__init__()

        # Lout = [  Lin + 2*padding - dilation * (kernel_size - 1) -1  ]+ 1
        #                               stride

        # conv
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        #relu
        self.relu = nn.ReLU()
        #maxpool
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #fc
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.drop = nn.Dropout(self.Dropout_r)

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
        print(f"Learning rate - {self.Learn_rate}")
        print(f"Numbers of Epochs - {self.Epoch_num}")
        print(f"Number of classes(numbers) - {self.Class_num}")
        print(f"Dropout rate - {self.Dropout_r}")
        print(f"Transform - {self.Transform}")

    def DataSetPrint(self):
        print(f"Shape of data Train - {self.dataset_train}")
        print(f"Shape of data Test - {self.dataset_test}")
        print(f"Shape of loader Train - {self.loader_train}")
        print(f"Shape of loader Test - {self.loader_test}")


    def ModelLayersPrint(self):
        print(f"conv1 - {self.conv1}")
        print(f"conv2 - {self.conv2}")
        print(f"fc1 - {self.fc1}")
        print(f"fc2 - {self.fc2}")
        print(f"drop - {self.drop}")
        print(f"relu - {self.relu}")
        print(f"pool - {self.pool}")
        
    def 
if __name__ == '__main__':
    #ImportsPrint()

    MCNN = MnistCNN()
    #MCNN.PreParamsPrint()
    #MCNN.ModelLayersPrint()
    MCNN.DataSetPrint()