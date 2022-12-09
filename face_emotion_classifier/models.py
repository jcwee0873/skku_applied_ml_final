import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34 ,ResNet34_Weights

class CNN(nn.Module):
    def __init__(self, cnn_output_size=1000):
        super(CNN, self).__init__()

        self.embedding = nn.Conv2d(1, 3, kernel_size=1, padding='same')
        self.cnn = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Linear(in_features=512, out_features=cnn_output_size, bias=True)
        
    def forward(self, x):
        e = self.embedding(x)
        z = self.cnn(e)
        
        return z
    
class Generator(nn.Module):
    def __init__(self, cnn_output_size=1000):
        super(Generator, self).__init__()
        
        self.fc = nn.Linear(
            in_features=cnn_output_size,
            out_features=4
        )
        
        self.activation = nn.LogSoftmax(dim=-1)
        
    def forward(self, z):
        y = self.fc(z)
        y = self.activation(y)
        
        return y
    
class FERClassifier(nn.Module):
    def __init__(self, cnn_output_size=1000):
        super(FERClassifier, self).__init__()
        
        self.cnn = CNN(cnn_output_size=cnn_output_size)
        self.generator = Generator(cnn_output_size=cnn_output_size)
        
    def forward(self, x):
        z = self.cnn(x)
        y = self.generator(z)
        
        return y, z