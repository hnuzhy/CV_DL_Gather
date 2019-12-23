#!/usr/bin/python3
# encoding: utf-8

import tensorflow as tf
from hrnet32 import HRNet_x1 as BackBone
from src.spm_config import spm_config as params

def SpmModel(inputs, num_joints, is_training = True):
    body = BackBone(inputs, training=is_training)
    body = head_net(body, 256, name='head', training=is_training)
    '''The original code does not notice that heatmap value should in range [0,1]'''
    # rootJoints = head_net(body, 1, name='root_joints', bn=False) 
    rootJoints = head_net(body, 1, name='root_joints', bn=False, activation='sigmoid')
    # displacement = head_net(body, 2*num_joints, name='reg_map', bn=False, activation='tanh')
    displacement = head_net(body, 2*num_joints, name='reg_map', bn=False)

    return rootJoints, displacement

def head_net(inputs, output_c, name='', bn=True, activation=None, training=True):
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01), name=name + '_conv3x3')(inputs)
    if bn:
        out = tf.keras.layers.BatchNormalization()(out, training=training)
    # Note: we can either use activation function in Conv2D layer or add another new activation function layer
    out = tf.keras.layers.ReLU(name=name + '_relu')(out)
    # :param --> activation: Activation function to use. If you don't specify anything, no activation is applied
    out = tf.keras.layers.Conv2D(filters=output_c, kernel_size=1, activation=activation,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01), name=name + '_conv1x1')(out)
    return out