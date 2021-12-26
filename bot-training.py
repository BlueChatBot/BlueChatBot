from utilization import tokenize, lowerStem, bagOfWords
import json 
import numpy as np

with open ('Information.json', 'r') as f:
   information = json.load(f)
#     print(information)

totalWords = [] 
tags = [] 
wordTag = []

for info in information['General']:
    tag = info['tag']
    tags.append(tag)  
    for pattern in info['patterns']: 
        placeholder = tokenize(pattern)  
        totalWords.extend(placeholder)  
        wordTag.append((placeholder, tag)) 

ignorePunc = ["'", '"', '.', '?', '!', ':', ';', ',']
all_words = [lowerStem(word) for word in totalWords if word not in ignorePunc]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

bag_training = []
tag_training = []

for tokenized_sentence, tag in wordTag:
    bag = bagOfWords(tokenized_sentence, all_words)
    bag_training.append(pattern)
    tag_label = tags.index(tag)
    tag_training.append(tag)

bag_training = np.array(bag_training)
tag_training = np.array(tag_training)

# Test 1: Tokenize
#string = "Oscar and Areeb are friends"
#print(string)
#string = tokenize(string)
#print(string)

# Test 2: Lowercase + stem words
#testwords = string 
#stemmed_words = [lowerStem(word) for word in testwords]
#print(stemmed_words)

