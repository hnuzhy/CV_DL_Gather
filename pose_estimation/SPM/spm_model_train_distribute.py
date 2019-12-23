import tensorflow as tf
from src.dataset import get_dataset
from src.spm_model import SpmModel
from src.spm_config import spm_config as params

import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu_ids']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':

    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Visible devices : ', visible_gpus)

    gpu_ids = [int(i) for i in params['gpu_ids'].split(',')]
    devices = ['/device:GPU:{}'.format(i) for i in gpu_ids]
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    cur_time = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M')
    checkpoint_prefix = os.path.join(params['ckpt'], cur_time)

    with strategy.scope():
        inputs = tf.keras.Input(shape=(params['height'], params['width'], 3), name='modelInput')
        outputs = SpmModel(inputs, num_joints=params['joints'], is_training=True)
        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.optimizers.Adam(learning_rate=3e-4)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)
        if params['finetune'] is not None:
            checkpoint.restore(params['finetune']).assert_existing_objects_matched()
            print('Successfully restore model from {}'.format(params['finetune']))
        dataset = get_dataset(len(gpu_ids))
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        print(dist_dataset.__dict__['_cloned_datasets'])


    with strategy.scope():
        def comput_loss(center_map, kps_map, preds):
            # L2Loss
            root_loss = tf.reduce_mean(tf.losses.mse(center_map, preds[0]))
            # SmoothL1Loss, HuberLoss
            weight = 1.0 - tf.cast(tf.equal(kps_map, 0), tf.float32)
            t = tf.abs(kps_map - pred[1] * weight)
            kps_loss = tf.reduce_mean(tf.where(t <= 1, 0.5 * t * t, 0.5 * (t - 1) ))
            per_example_loss = params['joint_weight']*kps_loss + params['root_weight']*root_loss
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=params['batch_size']*len(gpu_ids))
            
        def train_step(inputs):
            img, center_map, kps_map = inputs
            with tf.GradientTape() as tape:
                loss = comput_loss(center_map, kps_map, model(img))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        @tf.function
        def distribute_train_step(dataset_inputs):
            per_replica_loss = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)


        for epoch in range(params['total_epoch']):
            total_loss = 0.0
            for step, input in enumerate(dist_dataset):
                total_loss += distribute_train_step(input)
                if step % 20 == 0:
                    print('Epoch: {}, Train Steps: {}, Train Ave Loss: {}'.format(epoch, step, total_loss))
            checkpoint.save(checkpoint_prefix)

