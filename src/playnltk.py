from classiersfortext import TrainTieBot
from nltk.parse import generate
from random import choice
from nltk import pos_tag
import numpy as np
from nltk.corpus import wordnet as wn
from warnings import warn
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
def getSimilarWords(sentence, pos='VB'):
    ttb = TrainTieBot()
    similarWords = {}
    tokenDict = [ttb.tokenize(x) for x in sentence]
    wordFeatures = ttb.getFeaturesByName(tokenDict)
    for eachTuple in pos_tag(wordFeatures):
        if eachTuple[1].startswith(pos):
            for i, j in enumerate(wn.synsets(eachTuple[0])):
                similarWords[eachTuple[0]] = j.lemma_names()
            return similarWords


def generateSimilarSentences(sentence, pos='NN',  num_of_sentences=5):
    """ This function generates a similar sentence using the synonyms
    of word
    """
    sentences = []
    originalSentence = sentence.deepCopy()
    word_indices = dict()
    similarWords = getSimilarWords(sentence, pos)
    sentence = sentence.split('')
    temp = num_of_sentences
    for word, values in similarWords.iterItems():
        num_of_sentences = np.min(num_of_sentences, len(values[0]))
    if temp < num_of_sentences:
        warn('Number of available sentences is less than %s', temp)
    for eachWord in similarWords:
        word_indices[eachWord] = sentence.index(eachWord)
        count = 0
        while (count < num_of_sentences):
            newSentence = sentence



