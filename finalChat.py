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

# open information file
with open ('Information.json', 'r') as f:
    information = json.load(f)
# open data
File = "data.pth"
data = torch.load(File)

# retrieve all information
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# create model and train
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

chatBotName = "Blue ChatBot"
print(f"Welcome to {chatBotName}! Feel free to ask any question about the University of Michigan and I will try to answer it. Type 'all done' to exit!")

def chat(sentence):
    # while True: 
      #  sentence = input('You: ')
       # if sentence == "all done":
        #    break 

     # process sentence
    sentence = tokenize(sentence)
    # create bag of words
    BagOfWordsX = bagOfWords(sentence, all_words)
    BagOfWordsX = BagOfWordsX.reshape(1, BagOfWordsX.shape[0])
    BagOfWordsX = torch.from_numpy(BagOfWordsX).to(device)

    # get output 
    output = model(BagOfWordsX)
    _, predicted = torch.max(output, 1)
    # get predicted tag
    tag = tags[predicted.item()]

    #softmax
    # predict probability 
    probabilty = torch.softmax(output, dim = 1)
    actualProbability = probabilty[0][predicted.item()]
    # if certainty level is above 70% 
    if actualProbability > 0.70:
        for query in information["General"]:
            # select response corresponding to found tag
            if tag == query["tag"]:
                return(f"{chatBotName}: {random.choice(query['responses'])}")
                # if tag == "Goodbyes":
                # exit()
            
    # not over 70% confidence, reprompt
    else:
        return(f"{chatBotName}: I do not understand your question, can you please rephrase it?")