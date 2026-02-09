# импортеры
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

import cv2 as cv

class MnistCNN():

    dataset = MNIST(root = '/data', download = False, transform = transforms.ToTensor())
    data_train, data_valid = random_split(dataset, [50000, 10000])


    def Mtest(self):
        # beg
        #print(len(self.dataset))
        #print(len(self.data_train))
        #print(len(self.data_valid))
        #image, lable = self.dataset[10]
        #plt.imshow(image, cmap = 'gray')
        #plt.title("nya")
        #plt.show()

        # transform
         
        print(self.dataset)
        image_ten, lable = self.dataset[10]

        print(image_ten.shape, lable)
        plt.imshow(image_ten[0,10:15,10:15], cmap = 'gray')
        plt.title("nya")
        plt.show()



if __name__ == '__main__':
    MCNN = MnistCNN()
    MCNN.Mtest()