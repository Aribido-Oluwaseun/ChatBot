from classiersfortext import TrainTieBot
from nltk.parse import generate
from random import choice
from nltk import pos_tag
import numpy as np
from nltk.corpus import wordnet as wn

corpus =[{'1':"Hey hey hey let's go get lunch today?"},
             {'1': 'Hi have you had lunch'},
              {'2':'Did you go home'},
              {'3':'Hey!!! I need a favor'},
              {'1':'Hey lets go get a drink tonight'}]

query = ['are you going home']

GRAMMAR = """S -> NP VP
          S -> InterJ NP VP
          NP -> Det N
          NP -> Det N Det VP
          NP -> Det Adj N
          VP -> VTrans NP
          VP -> Vintr
          PP -> P NP 
          """
def getSimilarWords(word, pos='VB'):
    ttb = TrainTieBot()
    similarNouns = {}
    tokenDict = [ttb.tokenize(x) for x in corpus]
    wordFeatures = ttb.getFeaturesByName(tokenDict)
    for eachTuple in pos_tag(word):
        if eachTuple[1].startswith(pos):
            for i, j in enumerate(wn.synsets(eachTuple[0])):
                similarNouns[eachTuple[0]] = j.lemma_names()
            return similarNouns


def generateSimilarSentences(sentence, word, pos='NN',  num_of_sentences=5):
    """ This function generates a similar sentence using the synonyms
    of word
    """
    sentences = []
    similarwords = getSimilarWords(word, pos)
    sentence = sentence.split('')