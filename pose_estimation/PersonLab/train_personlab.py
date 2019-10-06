import os
import keras

from src.model import get_personlab
from src.tf_data_generator import *
from src.config import config
from src.polyak_callback import PolyakMovingAverage

from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras import backend as KB

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"

# Only allocate memory on GPUs as used
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
set_session(tf.Session(config=tf_config))

LOAD_MODEL_FILE = config.LOAD_MODEL_PATH
SAVE_MODEL_FILE = config.SAVE_MODEL_PATH

num_gpus = config.NUM_GPUS
batch_size_per_gpu = config.BATCH_SIZE_PER_GPU
batch_size = num_gpus * batch_size_per_gpu

input_tensors = get_data_input_tensor(batch_size=batch_size)
for i in range(len(input_tensors)):
    input_tensors[i].set_shape((None,)+input_shapes[i])

# Original paper uses intermediate_supervision to accelerate training.
if num_gpus > 1:
    with tf.device('/cpu:0'):
        model = get_personlab(train=True, input_tensors=input_tensors, 
            intermediate_supervision=config.INTER_SUPERVISION, intermediate_layer='res4b12_relu', with_preprocess_lambda=True)
else:
    model = get_personlab(train=True, input_tensors=input_tensors,
        intermediate_supervision=config.INTER_SUPERVISION, intermediate_layer='res4b12_relu', with_preprocess_lambda=True)

if LOAD_MODEL_FILE is not None and os.path.exists(LOAD_MODEL_FILE):
    model.load_weights(LOAD_MODEL_FILE, by_name=True)

if num_gpus > 1:
    parallel_model = multi_gpu_model(model, num_gpus)
else:
    parallel_model = model

# Custom loss layer is added into the model.py as the network output layer, 
# and the output of loss is used as the objective function of network optimization.
# When multiple losses are added, the Optimizer actually optimizes the sum of them.
for loss in parallel_model.outputs:
    parallel_model.add_loss(loss)
    
# In Keras, this metric will not be computed for this model, since the outputs have no targets.
# Only by commenting out that restriction in the Keras code will allow the display of these metrics
# which can be used to monitor the individual losses.
def identity_metric(y_true, y_pred):
    # Here, during training, y_pred is the average of all losses
    return KB.mean(y_pred)

def set_custom_callbacks():
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    history = LossHistory()  # print(history.losses)
    
    def save_model(epoch, logs):
        # You can save the list history.losses during this epoch
        save_model_path = SAVE_MODEL_FILE[:-3]+'_'+str(epoch).zfill(3)+'.h5'
        model.save_weights(save_model_path)
        
    callbacks = [LambdaCallback(on_epoch_end=save_model), history]

    if config.POLYAK:
        def build_save_model():
            with tf.device('/cpu:0'):
                save_model = get_personlab(train=True, input_tensors=input_tensors, 
                    intermediate_supervision=config.INTER_SUPERVISION, intermediate_layer='res4b12_relu', with_preprocess_lambda=True)
            return save_model
        polyak_save_path = '/'.join(config.SAVE_MODEL_FILE.split('/')[:-1]+['polyak_'+config.SAVE_MODEL_FILE.split('/')[-1]])
        polyak = PolyakMovingAverage(filepath=polyak_save_path, verbose=1, save_weights_only=True,
                                        build_model_func=build_save_model, parallel_model=True)
        callbacks.append(polyak)
    return callbacks
    
callbacks = set_custom_callbacks()

# The paper uses SGD optimizer with lr=0.0001, 0.001 is too large to train.
Kadam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
parallel_model.compile(target_tensors=None, loss=None, optimizer=Kadam, metrics=[identity_metric])
parallel_model.fit(steps_per_epoch=config.VAL_IMAGE_NUM//batch_size, epochs=config.NUM_EPOCHS, callbacks=callbacks)
# parallel_model.fit(steps_per_epoch=config.TRAIN_IMAGE_NUM//batch_size, epochs=config.NUM_EPOCHS, callbacks=callbacks)
