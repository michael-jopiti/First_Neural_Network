import torch
import torch.nn as nn
import torch.nn.functional as F 

#Firstly we need to create a model which inherits from NN
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=8, output_features=3):
        #initiate parent class (nn.Module)
        super().__init__()
        #Connect the different layers given the constructor parameters
            ## fc stands for fully connected and uses function nn.functional to connect all the different layers
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)
        
    def forward(self, x):
        # push from one layer to the other with the activation functional
        # in this example we're using relu which is Rectified Linear Unit
        # meaning that if x < 0 -> x = 0; else x = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
    
        return x

