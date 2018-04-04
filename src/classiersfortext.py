
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.lancaster import LancasterStemmer

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

    def updateCorpus(self, data):
        try:
            self.corpus.append(data)
            return True
        except (ValueError, NameError)as err:
            print ('Corpus not updated {}'.format(str(err)))

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

    def train(self, X, y, hidden_neurons=10, alpha=1, epochs=5000, dropout=False, dropout_percent=0.5):

        print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (
        hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(0)))
        np.random.seed(1)

        last_mean_error = 1
        # randomly initialize our weights with mean 0
        synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
        synapse_1 = 2 * np.random.random((hidden_neurons, len(0))) - 1

        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)
        """
        for j in iter(range(epochs + 1)):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0, synapse_0))

            if (dropout):
                layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                            1.0 / (1 - dropout_percent))

            layer_2 = sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j % 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print ("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                    break

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if (j > 0):
                synapse_0_direction_count += np.abs(
                    ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(
                    ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

            synapse_1 += alpha * synapse_1_weight_update
            synapse_0 += alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                   'datetime': now.strftime("%Y-%m-%d %H:%M"),
                   'words': words,
                   'classes': classes
                   }
        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)
        print ("saved synapses to:", synapse_file)
        """

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
            newCorpus.extend(eachList.values()[0])
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

    def run(self, corpus, query):
        corpus = self.tokenize(corpus)
        corpus = self.BOW(corpus)
        vectorize_corpus, vectorizer = self.vectorize(corpus)
        vectorize_query = vectorizer.transform(query).toarray()
        Y = range(0, vectorize_corpus.shape[0])
        clf = self.LogisticRegClassifier(vectorize_corpus, Y)
        print clf.predict(vectorize_query)

    def sigmoidDerivative(self, X):
        return X*(1-X)


def main():
    corpus =[{'1':"Hey hey hey let's go get lunch today?"},
             {'1': 'Hi have you had lunch'},
              {'2':'Did you go home'},
              {'3':'Hey!!! I need a favor'},
              {'4':'Hey lets go get a drink tonight'}]
    query = ['are you going home']
    #svmc = TrainTieBot()
    #svmc.BOW(corpus)
    #print svmc.BOWFit('When do you go home')


if __name__=='__main__':
    main()