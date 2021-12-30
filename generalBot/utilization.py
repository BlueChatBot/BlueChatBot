# import nltk
# nltk.download('punkt')
import re
import json
import pickle
import numpy as np

# from from nltk.stem.porter import 
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
mainStem = PorterStemmer()
# stop_words = set(stopwords.words("english"))

# Step 1: tokenize (break into individual words)
def tokenize(input_sentence): 
    return word_tokenize(input_sentence)
    
# Step 2: Lowercase + stem words
def lowerStem(word):
    return mainStem.stem(word.lower())

# Step 3: Create bag of words
def bagOfWords(tokenized_sentence, all_words):
    # further process by stemming and lowercasing word
    tokenized_sentence = [lowerStem(word) for word in tokenized_sentence]
    # create bag as array of length all words 
    bag = np.zeros(len(all_words), dtype = np.float32)
    # create array of 0, 1s
    # go through each idx and the word at that index
    for idx, word in enumerate(all_words):
        # check if the word in all words is in the tokenized sentence
        if word in tokenized_sentence:
            bag[idx] = 1
    return bag
    
    # LOGIC: 
    # all words ['hi','bye','lsa', 'eng', 'you', 'how', 'no', 'are']
    # tokenized sentence ['hi', 'how', 'are', 'you']
    # check if each word in all words list is in tokenized sentence
    # if it is then bag at that index is 1, else 0
    # in this case it would be: [1, 0, 0, 0, 1, 1, 0, 1]


#  def preprocess(input_sentence):
#     input_sentence = input_sentence.lower()
#     input_sentence = re.sub(r'[^\w\s]','',input_sentence)
#     tokens = tokenize(input_sentence)
#     input_sentence = [i for i in tokens if not i in stop_words]
#     return(input_sentence)