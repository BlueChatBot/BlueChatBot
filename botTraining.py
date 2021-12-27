from utilization import tokenize, lowerStem, bagOfWords
import json 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from model import NeuralNetwork
from torch.optim import SGD
DEBUG = 0; 
import numpy as np

# open information file and load content
with open ('Information.json', 'r') as f:
   information = json.load(f)
   
# create lists for words, tags, and word with tag 
totalWords = [] # list consisting of individual words 
tags = [] # list containing tags 
wordTag = [] # list containing tupple of words with their tag
 
# loop through each item of list 'General' (dictionaries) 
for info in information['General']:
    # store tag into variable
    tag = info['tag']
    # append tag into list of tags
    tags.append(tag)  
    # loop through each sentence in 'patterns' 
    for pattern in info['patterns']: 
        # tokenize sentence
        token = tokenize(pattern)  
        # add words into total words
        totalWords.extend(token) 
        # add tupple consisting of each word and its tag into word tag list 
        wordTag.append((token, tag)) 

# list of punctuations to ignore
ignorePunc = ["'", '"', '.', '?', '!', ':', ';', ',']
# list of all words processed in total words without punctuations
all_words = [lowerStem(word) for word in totalWords if word not in ignorePunc]
# sort the list of all words 
all_words = sorted(set(all_words))
# store sorted tags 
tags = sorted(set(tags))

# the bag of words after all preprocessing 
bag_training = []
# the tag sorted 
tag_training = []

# go through each tokenized pattern sentence and its tag in word tag list
for pattern_sentence, tag in wordTag:
    # create bag of words with pattern sentence and all words
    bag = bagOfWords(pattern_sentence, all_words)
    # add the bag into bag training 
    bag_training.append(bag)
    # get index of tag   
    tag_label = tags.index(tag)
    tag_training.append(tag_label)

# convert both into arrays (NEEDED?)
bag_training = np.array(bag_training)
tag_training = np.array(tag_training)

# class to activate and prepare data 
class chatData(Dataset): # inherit from class Dataset
    def __init__(self):
        self.number_samples = len(bag_training)
        self.bag_data = bag_training
        self.tag_data = tag_training
    
    def __getitem__(self, index):
        return self.bag_data[index], self.tag_data[index]

    def __len__(self):
        return self.number_samples

# set size and sample of data
batch_size = 8
hidden_size = 8

output_size = len(tags)
if DEBUG: print("OUTPUT SIZE: ", output_size)
input_size = len(bag_training[0])
learningRate = 0.001
num_epochs = 1000

# loading data 
dataset = chatData()
load_trainer = DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model (class defined in model.py)
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# loss optimizer
criteria = nn.CrossEntropyLoss()
# optimization function
optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
#optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)


for epoch in range(num_epochs):
    for (words, labels) in load_trainer:
        if DEBUG: print(words)
        words = words.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.int64)
        
        # clear the gradients
        optimizer.zero_grad()
        # forward pass
        # make prediction
        outputs = model(words)
        # calculate loss
        loss = criteria(outputs, labels)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
