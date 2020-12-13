NLP Chatbot


1) Theory + NLP Concepts (temming, toeknization, bag of words)
2) Create training data
3) Pytorch model and training
4) Save/ load model and training

Tokenizing Algorithm

import nltk
nltk.download('punkt') #It's pre-trained tokenizer alogoritm

from nltk.stem.porter import  PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
        return nltk.word_tokenize(sentence)

def stem(word):
        return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

a = "How long does shipping take?"
print(a)
a = tokenize(a)
print(a)

shiva@shivas-ubuntu:~/PycharmProjects/pythonProject/venv/src$ python3 nltk_utils.py
[nltk_data] Downloading package punkt to /home/shiva/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
How long does shipping take?
['How', 'long', 'does', 'shipping', 'take', '?']


Stemming
# Stemming
words = ["organise", "organizes", "organizing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

shiva@shivas-ubuntu:~/PycharmProjects/pythonProject/venv/src$ python3 nltk_utils.py
[nltk_data] Downloading package punkt to /home/shiva/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
['organis', 'organ', 'organ']



2. Creating the training Data

Load the Json file.

PyTorch Tutorial
https://www.youtube.com/watch?v=8qwowmiXANQ

Full tutorial
https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

https://github.com/python-engineer/pytorch-chatbot



