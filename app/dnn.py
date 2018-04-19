from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import dill

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


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
        self.export_dir_base = "/home/sbs/Desktop/Dev/ChatBot/EstimatorModels"
        np.random.seed(10)

    def serving_input_receiver_fn(self):
        serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
        receiver_tensors = {"predictor_inputs": serialized_tf_example}
        feature_spec = {"Question": tf.FixedLenFeature([128, 1], tf.string)}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    def create_input_function(self, df):
        # # Prediction on the test set.
        return tf.estimator.inputs.pandas_input_fn(df, df[self.labelKey], shuffle=True)

    def run(self, debug=False):
        init_op = tf.global_variables_initializer()

        # Reduce logging output.
        if debug:
            tf.logging.set_verbosity(tf.logging.ERROR)
        self.data_train.head()

        # Training input on the whole training set with no limit on training epochs.
        train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=self.data_train, y=self.data_train[self.labelKey], num_epochs=None, shuffle=True)

        # Prediction on the whole training set.
        predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        self.data_train, self.data_train[self.labelKey], shuffle=True)



        embedded_text_feature_column = hub.text_embedding_column(
            key=self.dataKey,
            module_spec=self.embeded_text_url)
        #print self.data_train
        #print len(np.unique(self.data_train[self.labelKey]))
        my_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs=1,  # Save checkpoints every 20 minutes.
            keep_checkpoint_max=13,  # Retain the 10 most recent checkpoints.
        )
        estimator = tf.estimator.DNNClassifier(
            hidden_units=self.hidden_units_size,
            feature_columns=[embedded_text_feature_column],
            n_classes=len(np.unique(self.data_train[self.labelKey])),
            optimizer=tf.train.AdagradOptimizer(learning_rate=self.learning_rate),
            model_dir=self.export_dir_base,
            config=my_checkpointing_config)

        # Training for 1,000 steps means 128,000 training examples with the default
        # batch size. This is roughly equivalent to 5 epochs since the training dataset
        # contains 25,000 examples.
        classifier = estimator.train(input_fn=train_input_fn, steps=100)

        # Save the training model
        #print('Exporting trained model to', self.export_dir_base)
        #classifier.export_savedmodel(export_dir_base=self.export_dir_base,
        #                               serving_input_receiver_fn=self.serving_input_receiver_fn, as_text=False)

        #
        train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
        # test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
        #
        # print "DNN Training set accuracy: {accuracy}".format(**train_eval_result)
        # print "DNN Test set accuracy: {accuracy}".format(**test_eval_result)
        return estimator

