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
        self.export_dir_base = "../EstimatorModels"

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
        self.data_train, self.data_train[self.labelKey], shuffle=False)

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

        train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
        test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

        print "Training set accuracy: {accuracy}".format(**train_eval_result)
        print "Test set accuracy: {accuracy}".format(**test_eval_result)


def serving_input_receiver_fn(input_data_df, labelKey):
  data = tf.estimator.inputs.pandas_input_fn(
        input_data_df, input_data_df[labelKey], num_epochs=None, shuffle=True)

  feature_spec = {'foo': tf.FixedLenFeature(['my name is joseph']),
                    'bar': tf.VarLenFeature('my name is joseph')}

  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[128, 1],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def main():
    value = serving_input_receiver_fn()
    print value

if __name__ == '__main__':
    main()