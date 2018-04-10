
import numpy as np
import pandas as pd
from dnn import DNN
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from warnings import warn
import copy
import json



class TrainTieBotException(Exception):
    "Defines error of logistic reg learner"


class CorpusException(Exception):
    "Defines error for the Corpus class"


class NeuralNetException(Exception):
    "Defined NeuralNet Exceptions"


class Corpus:

    def __init__(self, corpus=list()):
        if isinstance(corpus, list):
            self.corpus = corpus
        else:
            raise ('Corpus must be a list object')
        self.expandedSentences = []

    def updateCorpus(self, data):
        try:
            self.corpus.append(data)
            return True
        except (ValueError, NameError)as err:
            print ('Corpus not updated {}'.format(str(err)))

    @classmethod
    def load_data(cls, train_excel, test_excel):
        train_df = pd.read_excel(train_excel)
        test_df = pd.read_excel(test_excel)
        return train_df, test_df

    def deleteItem(self, item):
        if isinstance(item, str):
            for idx, object in enumerate(self.corpus):
                key, value = object.iterIterms()
                if value.lower() == item.lower():
                    self.corpus.pop(idx)
                return True
        else:
            raise CorpusException('Item to be deleted must be a question query!')

    def printCorpus(self):
        for object in self.corpus:
            print 'Index: {}, Question: {}'.format(object.keys()[0], object.values()[0])

    def getClasses(self):
        return list(set([int(x.keys()[0]) for x in self.corpus]))

    def getCorpus(self):
        return self.corpus

    def getSimilarWords(self, sentence, pos='NN'):
        ttb = TrainTieBot()
        similarWords = {}
        tokenDict = ttb.tokenize(sentence)
        wordFeatures = ttb.getFeaturesByName(tokenDict.values())
        for eachTuple in pos_tag(wordFeatures):
            if eachTuple[1].startswith(pos):
                for i, j in enumerate(wn.synsets(eachTuple[0])):
                    similarWords[str(eachTuple[0])] = [str(x) for x in j.lemma_names()]
                return similarWords, tokenDict.values()[0]

    def generateSimilarSentences(self, sentence, pos='NN', num_of_sentences=5):
        """ This function generates similar sentences using the synonyms
        of word depending on the part of speech passed
        The sentence argument must be a dict object.
        """
        assert (isinstance(sentence, dict))
        originalSentence = copy.deepcopy(sentence.values()[0])
        sentenceClass = sentence.keys()[0]
        newSentence = copy.deepcopy(sentence)
        similarWords = None
        tokens = None
        try:
            similarWords, tokens = self.getSimilarWords(newSentence, pos)
        except (TypeError) as err:
            #print('The Part of speech {} is not available'.format(pos))
            return None
        # Note that sentence becomes a string here
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

    def saveSentences(self, sentences, printSentence=False):
        if (sentences == []) or (sentences is None):
            pass
        elif len(sentences) > 1:
            for sent in sentences:
                self.saveSentences(sent)
        else:
            self.expandedSentences.append(sentences)

    def getExpandedSentences(self, corpus):
        for eachWord in corpus:
            pos = 'V'
            x = self.generateSimilarSentences(eachWord, pos)
            self.saveSentences(x)
        return self.expandedSentences


class SVMClassifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        return OneVsOneClassifier(SVC(
            C=1, cache_size=400, coef0=0.0,
            degree=5, gamma='auto', kernel='rbf',
            max_iter=-1, shrinking=True,
            tol=.01, verbose=False), -1).fit(self.X, self.y)

    def __call__(self):
        self.train()


class LogisticRegClassifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        return LogisticRegression().fit(self.X, self.y)

    def __call__(self):
        self.train()


class NeuralNet:

    def __init__(self, X, y):
        # check the input data to make sure they are in order
        if isinstance(X, pd.DataFrame):
            self.x = X
            self.y = y
        else:
            raise(NeuralNetException('X must be a pandas data frame'))
        classes = self.x.index.values

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidDerv(self, x):
        return x*(1-x)

    def train(self, X, y, hidden_neurons=50, alpha=1, epochs=10000, dropout=False, dropout_percent=0.5):
        print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (
        hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(0)))
        np.random.seed(1)
        output = np.zeros([1, len(y)])
        last_mean_error = 1
        # randomly initialize our weights with mean 0
        W_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
        W_1 = 2 * np.random.random((hidden_neurons, len(output))) - 1

        prev_W_0_weight_update = np.zeros_like(W_0)
        prev_W_1_weight_update = np.zeros_like(W_1)

        W_0_direction_count = np.zeros_like(W_0)
        W_1_direction_count = np.zeros_like(W_1)

        for j in iter(range(epochs + 1)):

            # Feed forward through layers 0, 1, and 2
            l_0 = X
            l_1 = self.sigmoid(np.dot(l_0, W_0))

            if (dropout):
                l_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                            1.0 / (1 - dropout_percent))

            l_2 = self.sigmoid(np.dot(l_1, W_1))

            # how much did we miss the target value?
            l_2_error = y - l_2

            if (j % int(epochs/5)) == 0 and j > epochs/10:
                # if this zzz iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(l_2_error)) < last_mean_error:
                    print ("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(l_2_error))))
                    last_mean_error = np.mean(np.abs(l_2_error))
                else:
                    print ("break:", np.mean(np.abs(l_2_error)), ">", last_mean_error)
                    break

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = l_2_error * self.sigmoidDerv(l_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(W_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * self.sigmoidDerv(l_1)

            W_1_weight_update = (l_1.T.dot(layer_2_delta))
            W_0_weight_update = (l_0.T.dot(layer_1_delta))

            if (j > 0):
                W_0_direction_count += np.abs(
                    ((W_0_weight_update > 0) + 0) - ((prev_W_0_weight_update > 0) + 0))
                W_1_direction_count += np.abs(
                    ((W_1_weight_update > 0) + 0) - ((prev_W_1_weight_update > 0) + 0))

            W_1 += alpha * W_1_weight_update
            W_0 += alpha * W_0_weight_update

            prev_W_0_weight_update = W_0_weight_update
            prev_W_1_weight_update = W_1_weight_update

        # persist synapses
        synapse = {'W0': W_0.tolist(), 'W1': W_1.tolist(),
                   'classes': output
                   }
        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)
        print ("saved synapses to:", synapse_file)


class TrainTieBot:
    def __init__(self):
        self.bagOfWords = None
        self.wordFeatures = None

    def tokenize(self, corpus):
        if not isinstance(corpus, dict):
            raise(TrainTieBotException('Corpus must be of type:dict()'))
        # tokenize each sentence
        tokenizer = RegexpTokenizer(r'\w+')
        for word in corpus:
            corpus[word] = corpus[word].lower()
            corpus[word] = tokenizer.tokenize(corpus[word])
        return corpus

    def cleanUpQuery(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        sentence = sentence.lower()
        return tokenizer.tokenize(sentence)

    def getFeaturesByName(self, corpus):
        if not isinstance(corpus, list):
            raise(TrainTieBotException('Corpus must be of type:list()'))
        newCorpus = list()
        for eachList in corpus:
            newCorpus.extend(eachList)
        corpus = newCorpus
        self.wordFeatures = list(set(corpus))
        return self.wordFeatures

    def BOW(self, corpus):
        if not isinstance(corpus, list):
            raise(TrainTieBotException('Corpus must be of type:list()'))
        tokenDict = [self.tokenize(x) for x in corpus]
        getWordFeatures = self.getFeaturesByName(tokenDict)
        vectorDataFrame = pd.DataFrame(data=np.zeros([len(corpus), len(getWordFeatures)]).astype(int))
        # add a label column
        labels = [x.keys()[0] for x in corpus]
        for word in vectorDataFrame.index.tolist():
            vectorDataFrame.loc[word, :] = \
                [corpus[word].values()[0].count(item) if item in corpus[word].values()[0] else 0 for item in getWordFeatures]
        vectorDataFrame['y'] = labels
        self.bagOfWords = vectorDataFrame
        return self.bagOfWords

    def BOWFit(self, query):
        if self.bagOfWords is None:
            raise (TrainTieBotException('Create Bag of Words vectors before fitting it to a new query.'))
        return [query.count(item) if item in query else 0 for item in self.wordFeatures]

    def list2df(self, data, dataKey, labelKey):
        data = {}
        data[dataKey] = []
        data[labelKey] = []
        for item in data:
            data[dataKey].append(data.values()[0])
            data[labelKey].append(data.keys()[0])
        return pd.DataFrame.from_dict(data)

    def df2list(self, df, key):
        df = df[key].to_dict()
        return [{x : str(df[x])} for x in df.keys()]

    def runDNN(self, query=None, corpus=None,):
        if corpus is None:
            train_excel = "/home/girija/Desktop/dev/TieTeam/ChatBot/QandAData.xlsx"
            test_excel = "/home/girija/Desktop/dev/TieTeam/ChatBot/test_data.xlsx"
            corpusObj = Corpus()
            corpusTrain, corpusTest = corpusObj.load_data(train_excel, test_excel)
            corpusTrain = self.df2list(corpusTrain, 'Question')
            corpusTest = self.df2list(corpusTest, 'Question')
            # Expand Vocabulary list with part of speeches
            corpusTrain = corpusObj.getExpandedSentences(corpusTrain)
            print corpusTrain
            corpusTest = corpusObj.getExpandedSentences(corpusTest)

            print "corpus test:"
            print corpusTest
            corpusTrain = self.list2df(corpusTrain, 'Question', 'y')
            corpusTest = self.list2df(corpusTest, 'Question', 'y')



        else:
            corpus = np.asarray(corpus)
            np.random.shuffle(corpus)
            corpusTrain = corpus
            corpusTest = np.asarray(query)
            corpusTrain = self.list2df(corpusTrain)
            corpusTest = self.list2df(corpusTest)

        dnnObject = DNN(pd_df_train=corpusTrain,
                        pd_df_test=corpusTest,
                        learning_rate=0.1,
                        hidden_units_size=[100, 100])
        dnnObject.run()


    def runSvm(self, corpus, query):
        pass

    def runLR(self, corpus, query):
        pass


def main():
    corpus =[{'1':"Hey hey hey let's go get lunch today?"},
             {'1':'Hi have you had lunch'},
             {'2':'Did you go home'},
             {'3':'Hey!!! I need a favor'},
             {'4':'Hey lets go get a drink tonight'},
             {'2':'Where is your home located?'},
             {'3':'Where can I find a favor'},
             {'4':'Drinks in Atlanta pubs are bad!'},
             {'1':'Hey did you bring lunch?'},
             {'4':'The drink is certainly bad'}]

    query = [{'2':'are you going home'}, {'3':'can you do me a favor?'}]
    tiebot = TrainTieBot()
    tiebot.runDNN()

if __name__=='__main__':
    main()