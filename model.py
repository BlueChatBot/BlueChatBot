import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
# from botTraining import ChatData 

# defining model for data
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = Linear(input_size, hidden_size) 
        self.l2 = Linear(hidden_size, hidden_size) 
        self.l3 = Linear(hidden_size, num_classes)
        self.relu = ReLU()  
    # forward propagation input
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out