## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        self.conv1      = nn.Conv2d(1, 16, 5,padding=4) 
        # now 224x224x16
        self.relu1      = F.relu
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool1   = nn.MaxPool2d(4,4)
        self.dropout1   = nn.Dropout2d(p=0.2)
        # now 56x56x16
        self.conv2      = nn.Conv2d(16,32,5,padding=4)
        self.relu2      = F.relu
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2   = nn.MaxPool2d(4,4)
        self.dropout2   = nn.Dropout2d(p=0.2)
        # now 14x14x32
        self.conv3      = nn.Conv2d(32,64,3, padding=2)
        self.relu3      = F.relu
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.maxpool3   = nn.MaxPool2d(2,2)
        self.dropout3   = nn.Dropout2d(p=0.2)
        # now 7x7x64
        self.fc4        = nn.Linear(4096,500)
        self.relu4      = F.relu
        self.batchnorm4 = nn.BatchNorm1d(500)
        self.dropout4   = nn.Dropout(p=0.2)
        self.fc5        = nn.Linear(500,136)       

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
              
        x = x.view(x.size(0), -1) 
        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class Net2(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        self.conv1      = nn.Conv2d(1, 16, 5,padding=4) 
        # now 224x224x16
        self.relu1      = F.relu
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool1   = nn.MaxPool2d(4,4)
        self.dropout1   = nn.Dropout2d(p=0.2)
        # now 56x56x16
        self.conv2      = nn.Conv2d(16,32,5,padding=4)
        self.relu2      = F.relu
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2   = nn.MaxPool2d(4,4)
        self.dropout2   = nn.Dropout2d(p=0.2)
        # now 14x14x32
        self.conv3      = nn.Conv2d(32,64,3, padding=2)
        self.relu3      = F.relu
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.maxpool3   = nn.MaxPool2d(2,2)
        self.dropout3   = nn.Dropout2d(p=0.2)
        # now 7x7x64
        self.fc4        = nn.Linear(4096,1000)
        self.relu4      = F.relu
        self.batchnorm4 = nn.BatchNorm1d(1000)
        self.dropout4   = nn.Dropout(p=0.2)
        self.fc5        = nn.Linear(1000,136)       

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
              
        x = x.view(x.size(0), -1) 
        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
