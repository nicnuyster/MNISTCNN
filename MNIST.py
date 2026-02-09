# импортеры
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

class MnistCNN():

    dataset = MNIST(root = '/data', download = False)
    data_train, data_valid = random_split(dataset, [50000, 10000])


    def MLog(self):
        print(len(self.dataset))
        print(len(self.data_train))
        print(len(self.data_valid))
        image, lable = self.dataset[10]
        plt.imshow(image, cmap = 'gray')
        plt.title("nya")
        plt.show()
        

    

if __name__ == '__main__':
    MCNN = MnistCNN()
    MCNN.MLog()