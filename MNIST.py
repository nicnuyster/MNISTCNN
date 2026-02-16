# импортеры
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import cv2 as cv

class MnistCNN():

    def __init__(self):
        print('\n' + " BEG __init__" + '\n')

        #my
        IMG_size = 28*28 # размер изображений
        CLASS_num = 10   # общ кол-во классов (цифры 0 - 9)

        dataset = MNIST(root = '/data', download = False, transform = transforms.ToTensor())
        data_train, data_valid = random_split(dataset, [50000, 10000])
        linear = nn.Linear(IMG_size, CLASS_num)

        batches = 128
        train_load = DataLoader(data_train, batches, shuffle = True)
        train_load = DataLoader(data_train, batches, shuffle = True)

        #tut
        super().__init__()
        self.linear = nn.Linear(self.IMG_size, self.CLASS_num)

        print('\n' + " END __init__" + '\n')


    def Mtest(self, printables = True):
        print('\n' + " BEG Mtest" + '\n')
        # beg

        #if printables == True: ***

        #print(len(self.dataset))
        #print(len(self.data_train))
        #print(len(self.data_valid))
        #image, lable = self.dataset[10]
        #plt.imshow(image, cmap = 'gray')
        #plt.title("nya")
        #plt.show()

        # transform

        #if printables == True: ***

        # print(self.dataset)
        # image_ten, lable = self.dataset[10]

        # print(image_ten.shape, lable)
        # plt.imshow(image_ten[0,10:15,10:15], cmap = 'gray')
        # plt.title("nya")
        # plt.show()

        print('\n' + " END Mtest" + '\n')

    # принтуем инфу о модели
    def ModelParramPrint(self, Printables = True):
        print('\n' + " BEG ModelParramPrint " + '\n')

        model = nn.Linear(self.IMG_size, self.CLASS_num)

        if Printables == True:
            print(model.weight.shape)
            #print(model.weight)
            print(model.bias.shape)
            #print(model.bias)

        print('\n' + " END ModelParramPrint " + '\n')

    def ForwardPass(self, xb, Printables = True):
        print('/n' + " BEG ForwardPass" + '/n')

        #if printables == True: 

        xb = xb.reshape(-1, 784)
        print(xb)
        out = self.linear(xb)
        print(out)
        print('/n' + " END ForwardPass" + '/n')

        return(out)
    
    def TrainingStep(self, batch):
        print('/n' + " BEG TrainingStep" + '/n')

        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)

        print('/n' + " END trainingStep" + '/n')
        return({'val_loss': loss, 'val_acc': acc})

    def LearningStep(self, batch):
        print('/n' + " BEG LearningStep" + '/n')

        print('/n' + " END LearningStep" + '/n')

    def ValidationStep(self, batch):
        print('/n' + " BEG ValidationStep" + '/n')

        print('/n' + " END ValidationStep" + '/n')

if __name__ == '__main__':
    MCNN = MnistCNN()
    MCNN.Mtest()
    MCNN.ModelParramPrint()