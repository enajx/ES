import torch
import torch.nn as nn


        
class MLP(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space_dim, action_space_dim, bias=False):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_space_dim, 128, bias=bias) 
        self.linear2 = nn.Linear(128, 64, bias=bias)
        self.out = nn.Linear(64, action_space_dim, bias=bias)

    def forward(self, ob):
        state = torch.as_tensor(ob).float().detach()    
        x = torch.tanh(self.linear1(state))   
        x = torch.tanh(self.linear2(x))
        o = self.out(x)
        return o.squeeze()

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters()).detach().numpy()
    
    
class CNN(nn.Module):
    "CNN+MLP"
    def __init__(self, input_channels, action_space_dim):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, stride=1, bias=False)   
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=False)
        
        self.linear1 = nn.Linear(648, 128, bias=False) 
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.out = nn.Linear(64, action_space_dim, bias=False)

    def forward(self, ob):
        
        state = torch.as_tensor(ob.copy())
        state = state.float()
        
        x = self.pool(torch.tanh(self.conv1(state)))
        x = self.pool(torch.tanh(self.conv2(x)))
        
        x = x.view(-1)
        
        x = torch.tanh(self.linear1(x))   
        x = torch.tanh(self.linear2(x))
        o = self.out(x)
        return o

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters() ).detach().numpy()