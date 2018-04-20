
import numpy as np
import pandas as pd
from dnn import DNN
import restoreModel
from flask import Flask, render_template
from flask import sessions
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from warnings import warn
import copy



class TrainTieBotException(Exception):
    "Defines error of logistic reg learner"


class CorpusException(Exception):
    "Defines error for the Corpus class"


class NeuralNetException(Exception):
    "Defined NeuralNet Exceptions"

TRAIN_EXCEL = "/home/sbs/Desktop/Dev/ChatBot/QandAData.xlsx"
TEST_EXCEL = "/home/sbs/Desktop/Dev/ChatBot/test_data.xlsx"

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
    def load_data(cls, dataLoaction):
        df = pd.read_excel(dataLoaction)
        return df

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
        sentClass = sentence.keys()[0]
        ttb = TrainTieBot()
        similarWords = {}
        tokenDict = ttb.tokenize(sentence)

        wordFeatures = ttb.getFeaturesByName([{sentClass: tokenDict}])
        for eachTuple in pos_tag(wordFeatures):
            if eachTuple[1].startswith(pos):
                for i, j in enumerate(wn.synsets(eachTuple[0])):
                    similarWords[str(eachTuple[0])] = [str(x) for x in j.lemma_names()]
                return similarWords, tokenDict

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

        # Note that sentence becomes a string here
        sentences = list()
        word_indices = dict()

        try:
            similarWords, tokens = self.getSimilarWords(newSentence, pos)
        except (TypeError, ValueError) as err:
            #print('The Part of speech {} is not available'.format(pos))
            return {sentenceClass: originalSentence}

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
        if sentences == []:
            sentences.append({sentenceClass: originalSentence})
        return sentences

    def saveSentences(self, sentences, printSentence=False):
        if (sentences == []) or (sentences is None):
            pass
        elif len(sentences) > 1:
            for sent in sentences:
                self.saveSentences(sent, printSentence)
        elif (len(sentences) == 1) and isinstance(sentences, list):
            if printSentence:
                print sentences[0]
            self.expandedSentences.append(sentences[0])
        else:
            if printSentence:
                print sentences
            self.expandedSentences.append(sentences)

    def getExpandedSentences(self, corpus, debug=False):
        for eachWord in corpus:
            pos = 'V'
            x = self.generateSimilarSentences(eachWord, pos)
            self.saveSentences(x, debug)
        return self.expandedSentences


class SVMClassifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        return OneVsOneClassifier(SVC(
            C=0.75, cache_size=400, coef0=0.0,
            degree=5, gamma='auto', kernel='rbf',
            max_iter=-1, shrinking=True,
            tol=.01, verbose=False), -1).fit(self.X, self.y)



class LogisticRegClassifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        return LogisticRegression().fit(self.X, self.y)



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

class TrainTieBot:

    def __init__(self):
        self.bagOfWords = None
        self.wordFeatures = None

    def tokenize(self, corpus):
        if not isinstance(corpus, dict):
            raise(TrainTieBotException('Corpus must be of type:dict()'))
        # # tokenize each sentence

        tokenizer = RegexpTokenizer(r'\w+')
        token = [str(x) for x in tokenizer.tokenize(corpus.values()[0].lower())]
        return token


    def cleanUpQuery(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        sentence = sentence.lower()
        return tokenizer.tokenize(sentence)

    def getFeaturesByName(self, corpus):
        words = list()
        if not isinstance(corpus, list):
            raise(TrainTieBotException('Corpus must be of type:list()'))
        for x in corpus:
            words.extend(x.values()[0])
        self.wordFeatures = list(set(words))
        return self.wordFeatures

    def BOW(self, corpus):
        if not isinstance(corpus, list):
            raise(TrainTieBotException('Corpus must be of type:list()'))
        tokenDict = [{x.keys()[0]: self.tokenize(x)} for x in corpus]
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
        tokenDict = [{x.keys()[0]: self.tokenize(x)} for x in query]
        arrayFitDf = pd.DataFrame(data=np.zeros([len(tokenDict), len(self.wordFeatures)])).astype(int)
        arrayFitDf['y'] = [x.keys()[0] for x in query]
        for i in range(len(query)):
            arrayFitDf.iloc[i, :-1] = [query[i].values()[0].count(item) if item in query[i].values()[0] else 0 for item in self.wordFeatures]
        return arrayFitDf

    def list2df(self, data, dataKey, labelKey):
        df = {}
        df[dataKey] = []
        df[labelKey] = []
        for obj in data:
            df[dataKey].append(obj.values()[0])
            df[labelKey].append(obj.keys()[0])
        return pd.DataFrame.from_dict(df)

    def df2list(self, df, dataKey, labelKey):
        df = df[[dataKey, labelKey]]
        return [{df[labelKey][x]:df[dataKey][x]} for x in range(df.shape[0])]

    def prepareDF(self, query=None, corpus=None, dataKey='Question', labelKey='y', excelLocation=None):
        if corpus is None:
            corpusObj = Corpus()
            corpusT = corpusObj.load_data(excelLocation)
            corpusT = self.df2list(corpusT, dataKey, labelKey)
            # Expand Vocabulary list with part of speeches
            corpusT = corpusObj.getExpandedSentences(corpusT)
            # #print "corpus test:"
            corpusT = self.list2df(corpusT, dataKey, labelKey)
        else:
            corpusObj = Corpus()
            corpusT = corpusObj.getExpandedSentences(corpus)
            corpusT = self.list2df(corpusT, dataKey, labelKey)
        return corpusT


    def getXandY(self, corpus=None, query=None):
        corpusObj = Corpus()
        dataKey = 'Question'
        labelKey = 'y'
        corpusTrain, corpusTest = corpusObj.load_data(TRAIN_EXCEL, TEST_EXCEL)
        corpusTrain = self.df2list(corpusTrain, dataKey, labelKey)
        corpusTest = self.df2list(corpusTest, dataKey, labelKey)
        corpusTrain = corpusObj.getExpandedSentences(corpusTrain)
        corpusTestObj = Corpus()
        corpusTest = corpusTestObj.getExpandedSentences(corpusTest)
        BOWTrain = self.BOW(corpusTrain)
        BOWTest = self.BOWFit(corpusTest)

        X = BOWTrain.iloc[:, :-1]
        y = BOWTrain['y']
        X_test = BOWTest.iloc[:, :-1]
        y_test = BOWTest['y']
        return X, y, X_test, y_test

    def runDNNTrain(self, learningRate=0.01, hiddenUnitSize=[50, 100], dataKey='Question', labelKey='y'):
        corpusTrain = self.prepareDF(excelLocation=TRAIN_EXCEL)

        dnnObject = DNN(pd_df_train=corpusTrain,
                        pd_df_test=None,
                        learning_rate=learningRate,
                        hidden_units_size=hiddenUnitSize,
                        dataKey=dataKey,
                        labelKey=labelKey)
        result = dnnObject.run()
        return result

    def runDNNPredict(self):
        XTestDf = self.prepareDF(excelLocation=TEST_EXCEL)
        restoreModel.predict(X_test=XTestDf)


    def runSVM(self, corpus=None, query=None):
        X, y, X_test, y_test = self.getXandY()
        svm = SVMClassifier(X, y)
        clf = svm.train()
        y_pred = clf.predict(X)
        y_test_pred = clf.predict(X_test)
        result1 = float(sum((y == y_pred) + 0)) / y_pred.shape[0]
        result2 = float(sum((y_test == y_test_pred) + 0)) / y_test_pred.shape[0]
        print 'SVM Training set accuracy: ', result1
        print 'SVM Test set accuracy: ', result2

    def runLR(self, corpus=None, query=None):
        X, y, X_test, y_test = self.getXandY()
        lr = LogisticRegClassifier(X, y)
        clf = lr.train()
        y_pred = clf.predict(X)
        y_test_pred = clf.predict(X_test)
        result1 = float(sum((y == y_pred) + 0))/y_pred.shape[0]
        result2 = float(sum((y_test == y_test_pred) + 0))/y_test_pred.shape[0]
        print 'LR Training set accuracy: ', result1
        print 'LR Test set accuracy: ', result2


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
    #tiebot.runLR()
    #tiebot.runSVM()
    #tiebot.runDNNTrain()
    tiebot.runDNNPredict()


if __name__=='__main__':
    main()