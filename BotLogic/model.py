import torch.nn as nn
from torch.nn import Linear, ReLU
# from botTraining import ChatData 

# defining model for data
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        # defining multiple layers
        self.l1 = Linear(input_size, hidden_size)  # single layer feedforward network
        self.l2 = Linear(hidden_size, hidden_size) 
        self.l3 = Linear(hidden_size, num_classes)
        # activation function (output 0 if number is negative, and number if it is positive)
        self.relu = ReLU()
    # forward propagation input
    def forward(self, BagOfWordsX):
        # pass through multiple layers and make adjustments when necessary
        out = self.l1(BagOfWordsX)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out