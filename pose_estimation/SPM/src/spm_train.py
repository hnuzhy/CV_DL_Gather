#!/usr/bin/python3
# encoding: utf-8

import tensorflow as tf
from losses import spm_loss
from spm_config import spm_config as params

import os
import time
import numpy as np

@tf.function
def infer(model, inputs):
    preds = model(inputs)
    return preds

def train(model, optimizer, dataset, epochs, cur_time='8888-88-88-88'):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    if params['finetune'] is not None:
        manager = tf.train.CheckpointManager(ckpt, params['finetune'], max_to_keep=3)
        ckpt.restore(params['finetune'])
        print('successfully restore model from ... {}'.format(params['finetune']))
    else:
        manager = tf.train.CheckpointManager(ckpt, os.path.join(params['ckpt'], cur_time), max_to_keep=3)

    for epoch in range(epochs):
        tic_epoch = time.time()
        # for step, (img, center_map, kps_map, kps_map_weight) in enumerate(dataset):
        for step, (img, center_map, kps_map) in enumerate(dataset):
            with tf.GradientTape() as tape:
                preds = infer(model, img)
                # loss = spm_loss(center_map, kps_map, kps_map_weight, preds)
                loss = spm_loss(center_map, kps_map, preds)
            grads = tape.gradient(loss[0], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ckpt.step.assign_add(1)
            if step % 10 == 0:
                tf.summary.scalar('loss', loss[0], step=int(ckpt.step))
                tf.summary.scalar('root_joint_loss', loss[1], step=int(ckpt.step))
                tf.summary.scalar('offset_loss', loss[2], step=int(ckpt.step))
            if step % 100 == 0:
                gt_root_joints = center_map
                pred_root_joints = preds[0]
                tf.summary.image('gt_root_joints', gt_root_joints, step=int(ckpt.step), max_outputs=8)
                tf.summary.image('pred_root_joints',pred_root_joints, step=int(ckpt.step), max_outputs=8)
                tf.summary.image('img', img, step=int(ckpt.step), max_outputs=8)
                print('Epoch [{}] step [{}] -- \t Total_Loss : {}, \t Root_joint_Loss : {}, \t Body_joint_Loss : {}'.format(
                    epoch, step, np.round(loss[0], 3), np.round(loss[1], 3), np.round(loss[2], 3)))
                print("Until step {} in epoch {} takes time: {}".format(step, epoch, np.round(time.time()-tic_epoch, 3)))
        save_path = manager.save()
        print('Saved ckpt for step {} : {}'.format(int(ckpt.step), save_path))
        print("Epoch {} takes time: {}".format(epoch, np.round(time.time()-tic_epoch, 3)))