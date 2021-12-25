from utilization import tokenize, lowerStem
import json

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

print(wordTag) 
# Test 1: Tokenize
string = "Oscar and Areeb are friends"
print(string)
string = tokenize(string)
print(string)

# Test 2: Lowercase + stem words
testwords = string 
stemmed_words = [lowerStem(word) for word in testwords]
print(stemmed_words)

