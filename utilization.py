import nltk
nltk.download('punkt')
import re
import json
import pickle
import numpy as np

# from from nltk.stem.porter import 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
mainStem = PorterStemmer()
# stop_words = set(stopwords.words("english"))

# Step 1: Read in the data//tokenize
def tokenize(input_sentence): 
    return nltk.word_tokenize(input_sentence)
    
# Step 2: Lowercase + stem words
def lowerStem(word):
    return mainStem.stem(word.lower())

# Step 3: Create bag of words
def bagOfWords(tokenized_sentence, all_words):
    pass
    tokenized_sentence = [lowerStem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words, dtype = np.float32))

#  def preprocess(input_sentence):
#     input_sentence = input_sentence.lower()
#     input_sentence = re.sub(r'[^\w\s]','',input_sentence)
#     tokens = tokenize(input_sentence)
#     input_sentence = [i for i in tokens if not i in stop_words]
#     return(input_sentence)