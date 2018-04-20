from classiersfortext import TrainTieBot
from nltk.parse import generate
from random import choice
from nltk import pos_tag
import numpy as np
from nltk.corpus import wordnet as wn
from warnings import warn
import copy

corpus =[{'1':"How to perform PCA Test"},
             {'2': 'What is an App Server?'},
              {'3':'What is Command Center'},
              {'3':'How do I generate SeedFiles'},
              {'4':'How to install RadioShop'}]

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
def getSimilarWords(sentence, pos):
    ttb = TrainTieBot()
    similarWords = {}
    tokenDict = ttb.tokenize(sentence)
    wordFeatures = ttb.getFeaturesByName(tokenDict.values())
    for eachTuple in pos_tag(wordFeatures):
        if eachTuple[1].startswith(pos):
            for i, j in enumerate(wn.synsets(eachTuple[0])):
                similarWords[str(eachTuple[0])] = [str(x) for x in j.lemma_names()]
            return similarWords, tokenDict.values()[0]


def generateSimilarSentences(sentence, pos,  num_of_sentences=5):
    """ This function generates similar sentences using the synonyms
    of word depending on the part of speech passed
    The sentence argument must be a dict object.
    """
    assert(isinstance(sentence, dict))
    originalSentence = copy.deepcopy(sentence.values()[0])
    sentenceClass = sentence.keys()[0]
    newSentence = copy.deepcopy(sentence)
    similarWords = None
    tokens = None
    try:
        similarWords, tokens = getSimilarWords(newSentence, pos)
    except (TypeError) as err:
        print('The Part of speech {} is not available'.format(pos))
        return None
    #Note that sentence becomes a string here
    sentences = list()
    word_indices = dict()
    temp = num_of_sentences


    for word, values in similarWords.iteritems():
        num_of_sentences = np.min([num_of_sentences, len(values)])
        word_indices[word] = tokens.index(word)


    if temp < num_of_sentences:
        warn('Number of available sentences is less than %s', temp)
    for eachWord in similarWords:
        count = 0
        while (count < num_of_sentences):
            sentence = tokens
            sentence[word_indices[eachWord]] = str(similarWords[eachWord][count])
            someSentence = ' '.join(sentence)
            sentences.append({sentenceClass: someSentence})
            count += 1
    return sentences


def printSentences(sentences):
    if (sentences == []) or (sentences is None):
        pass
    elif len(sentences) > 1:
        for sent in sentences:
            printSentences(sent)
    else:
        print sentences

for eachWord in corpus:
    pos = 'V'
    x = generateSimilarSentences(eachWord, pos)
    printSentences(x)


