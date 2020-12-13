import torch
import nltk
nltk.download('punkt') #It's pre-trained tokenizer alogoritm
import numpy as np

from nltk.stem.porter import  PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
        return nltk.word_tokenize(sentence)

def stem(word):
        return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for indx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[indx] = 1.0
            
    return bag
    
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)
# Output - [0. 1. 0. 1. 0. 0. 0.]

# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)

# Stemming
# words = ["organise", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)