import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import pandas as pd
import numpy as np

def predict(X_test):
    init_op = tf.global_variables_initializer()
    dataKey = 'Question'
    labelKey = 'y'
    features = X_test.to_dict()
    labels = {labelKey: features[labelKey]}
    features = {dataKey: features[dataKey]}
    full_model_dir = "/home/sbs/Desktop/Dev/ChatBot/EstimatorModels"
    tf.reset_default_graph()
    with tf.Session() as sess:
        file_name = tf.train.latest_checkpoint(full_model_dir, latest_filename=None) + '.meta'
        saver = tf.train.import_meta_graph(file_name)
        saver.restore(sess, tf.train.latest_checkpoint(full_model_dir))

