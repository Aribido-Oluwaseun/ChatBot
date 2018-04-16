import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re



class DNN:

    def __init__(self, pd_df_train=None,
                 pd_df_test=None,
                 hidden_units_size=[500, 100],
                 embedded_text_url=None,
                 learning_rate=0.003,
                 dataKey='Question',
                 labelKey='y'):
        self.data_train = pd_df_train
        self.data_test = pd_df_test
        if embedded_text_url is None:
            self.embeded_text_url = "https://tfhub.dev/google/nnlm-en-dim128/1"
        else:
            self.embeded_text_url = embedded_text_url
        self.hidden_units_size = hidden_units_size
        self.learning_rate = learning_rate
        self.dataKey = dataKey
        self.labelKey = labelKey

    def run(self):
        # Reduce logging output.
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.data_train.head()
        # Training input on the whole training set with no limit on training epochs.
        train_input_fn = tf.estimator.inputs.pandas_input_fn(
        self.data_train, self.data_train[self.labelKey], num_epochs=None, shuffle=True)

        # Prediction on the whole training set.
        predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        self.data_train, self.data_train[self.labelKey], shuffle=False)

        # Prediction on the test set.
        predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        self.data_test, self.data_test[self.labelKey], shuffle=False)

        embedded_text_feature_column = hub.text_embedding_column(
            key=self.dataKey,
            module_spec=self.embeded_text_url)
        #print self.data_train
        #print len(np.unique(self.data_train[self.labelKey]))
        estimator = tf.estimator.DNNClassifier(
            hidden_units=self.hidden_units_size,
            feature_columns=[embedded_text_feature_column],
            n_classes=len(np.unique(self.data_train[self.labelKey])),
            optimizer=tf.train.AdagradOptimizer(learning_rate=self.learning_rate))

        # Training for 1,000 steps means 128,000 training examples with the default
        # batch size. This is roughly equivalent to 5 epochs since the training dataset
        # contains 25,000 examples.
        estimator.train(input_fn=train_input_fn, steps=1000)
        classes=[]
        for x in  (estimator.predict(predict_test_input_fn, predict_keys="classes")):
            classes.append(x)

        train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
        test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

        print "Training set accuracy: {accuracy}".format(**train_eval_result)
        print "Test set accuracy: ", test_eval_result

        return classes

