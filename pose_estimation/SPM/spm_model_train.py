#!/usr/bin/python3
# encoding: utf-8

# Follow the author's personal blog https://niexc.github.io/ to wait for source code releasing

import tensorflow as tf
from src.dataset import get_dataset
from src.spm_model import SpmModel
from src.spm_config import spm_config as params
from src.spm_train import train

import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':

    inputs = tf.keras.Input(shape=(params['height'], params['width'], 3), name='modelInput')
    outputs = SpmModel(inputs, num_joints=params['joints'], is_training=True)
    model = tf.keras.Model(inputs, outputs)

    cur_time = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M')

    optimizer = tf.optimizers.Adam(learning_rate=params['learning_rate'])
    dataset = get_dataset()
    summary_writer = tf.summary.create_file_writer(os.path.join('./spm_logs', cur_time))
    with summary_writer.as_default():
        train(model, optimizer, dataset, params['total_epoch'], cur_time)

