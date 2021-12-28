from utilization import tokenize, lowerStem, bagOfWords
import json 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from model import NeuralNetwork
from torch.optim import SGD
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open ('Information.json', 'r') as f:
    information = json.load(f)

File = "data.pth"
data = torch.load(File)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

chatBotName = "Blue ChatBot"
print("Welcome to Blue ChatBot! Feel free to ask any question about the University of Michigan and I will try to answer it. Type 'all done' to exit!")

while True: 
    sentence = input('You: ')
    if sentence == "all done":
        break 
    sentence = tokenize(sentence)
    BagOfWordsX = bagOfWords(sentence, all_words)
    BagOfWordsX = BagOfWordsX.reshape(1, BagOfWordsX.shape[0])
    BagOfWordsX = torch.from_numpy(BagOfWordsX).to(device)

    output = model(BagOfWordsX)
    _, predicted = torch.max(output, 1)
    tag = tags[predicted.item()]

    #softmax
    probabilty = torch.softmax(output, dim = 1)
    actualProbability = probabilty[0][predicted.item()]
    if actualProbability > 0.70:
        for query in information["General"]:
            if tag == query["tag"]:
                print(f"{chatBotName}: {random.choice(query['responses'])}")
    else:
        print(f"{chatBotName}: I do not understnand your question, can you please rephrase it?")