from generalBot.utilization import tokenize, lowerStem, bagOfWords
import json 
import numpy as np

# Test 1: Tokenize
string = "Oscar and Areeb are friends"
print(string)
string = tokenize(string)
print(string)

# Expected: ['Oscar', 'and', 'Areeb', 'are', 'friends']

# Test 2: Lowercase + stem words
testwords = string 
stemmed_words = [lowerStem(word) for word in testwords]
print(stemmed_words)
# expected: ['oscar', 'and', 'areeb', 'are', 'friend']

# Test 3: Create bag of words
test_sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "are", "thank you", "doing", "good", "bad", "great"]
bag = bagOfWords(test_sentence, words)
print(bag)
# expected [1, 0, 1, 1, 1, 0, 0]