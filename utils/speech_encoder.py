'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''

import tensorflow as tf
from utils.ops import fc_layer, conv2d, BatchNorm

class SpeechEncoder:
    def __init__(self, config, scope='SpeechEncoder'):
        self.scope = scope
        self._speech_encoding_dim = config['expression_dim']
        #self._condition_speech_features = config['condition_speech_features']
        self._speech_encoder_size_factor = config['speech_encoder_size_factor']

    def __call__(self, speech_features, is_training, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()

            batch_norm = BatchNorm(epsilon=1e-5, momentum=0.9)
            speech_features = batch_norm(speech_features, reuse=reuse, is_training=is_training)

            speech_feature_shape = speech_features.get_shape().as_list()
            speech_features_reshaped = tf.reshape(tensor=speech_features, shape=[-1, speech_feature_shape[1]])

            concatenated = speech_features_reshaped

            units_in = concatenated.get_shape().as_list()[1]

            with tf.name_scope('fc0'):
                fc0 = tf.nn.tanh(fc_layer(concatenated, num_units_in=units_in, num_units_out=128, scope='fc0'))

            with tf.name_scope('fc00'):
                fc00 = tf.nn.tanh(fc_layer(fc0, num_units_in=128, num_units_out=128, scope='fc00'))

            with tf.name_scope('fc1'):
                fc1 = tf.nn.tanh(fc_layer(fc00, num_units_in=128, num_units_out=128, scope='fc1'))
            with tf.name_scope('fc2'):
                fc2 = tf.nn.tanh(fc_layer(fc1, num_units_in=128, num_units_out=self._speech_encoding_dim, scope='fc2'))
            return fc2
