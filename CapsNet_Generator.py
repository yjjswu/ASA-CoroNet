from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

# import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import math
from itertools import cycle
import numpy as np
import pandas as pd
# import seaborn as sn
from matplotlib import pyplot as plt

from evaluation import compute_performance_measures
from utils import *
from layers import *

K.set_image_data_format('channels_last')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 按需分配显存
# keras.backend.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
tf.random.set_seed(111111)
np.random.seed(111111)
random.seed(111111)

tf.__version__

# %%

import tensorflow as tf

# %%

tf.__version__

# %%


tf.test.gpu_device_name()

# %%

from tensorflow.python.client import device_lib

device_lib.list_local_devices()


# %%


# %%

def capsnet(inputs):
    x = Conv2D(16, 5, 1, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                           gamma_initializer="random_uniform")(x)
    x = ReLU()(x)

    x = MaxPooling2D()(x)

    x = Conv2D(32, 5, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                           gamma_initializer="random_uniform")(x)
    x = LeakyReLU()(x)

    x = MaxPooling2D()(x)

    x = Conv2D(64, 5, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                           gamma_initializer="random_uniform")(x)
    x = LeakyReLU()(x)

    x = MaxPooling2D()(x)

    x = Conv2D(128, 5, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                           gamma_initializer="random_uniform")(x)
    x = LeakyReLU()(x)

    x = PrimaryCaps_H(32, 8, 9, 1, padding='SAME')(x)

    digit_caps = DigitCaps(3, 16, 3)(x)

    digit_caps_len = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(digit_caps)
    model = Model(inputs=[inputs], outputs=[digit_caps, digit_caps_len], name='CapsNet')

    return model


# %%

def generator(input_shape):
    inputs = Input(16 * 3)

    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)

    return Model(inputs=[inputs], outputs=[x], name='Generator')


# %%

class Mask(Layer):
    def call(self, inputs, double_mask=None, **kwargs):
        if type(inputs) is list:
            if double_mask:
                inputs, mask1, mask2 = inputs
            else:
                inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            if double_mask:
                mask1 = tf.keras.backend.one_hot(tf.argsort(x, direction='DESCENDING', axis=-1)[..., 0],
                                                 num_classes=x.get_shape().as_list()[1])
                mask2 = tf.keras.backend.one_hot(tf.argsort(x, direction='DESCENDING', axis=-1)[..., 1],
                                                 num_classes=x.get_shape().as_list()[1])
            else:
                mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        if double_mask:
            masked1 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask1, -1))
            masked2 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask2, -1))
            return masked1, masked2
        else:
            masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
            return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # generation step
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


# %%

def create_model(inputs, y_true, mode='train'):
    capsnet_model = capsnet(inputs)
    digit_caps, digit_caps_len = capsnet_model(inputs)

    # 重构
    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)

    generator_model = generator([128, 128, 3])
    x_gen_train = generator_model(masked_by_y)
    x_gen_eval = generator_model(masked)

    if mode == 'train':
        return Model([inputs, y_true], [digit_caps_len, x_gen_train], name='CapsNet_Generator')
    elif mode == 'test':
        return Model(inputs, [digit_caps_len, x_gen_eval], name='CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')


# %%

inputs = Input(shape=(128, 128, 3))
y_true = Input(shape=(3,))

model = create_model(inputs, y_true)

# %%

adam = optimizers.Adam(lr=0.0001)

model.compile(loss=[margin_loss, 'mse'], optimizer=adam, loss_weights=[1., 0.392], metrics={'CapsNet': 'accuracy'})
model.summary()

# %%

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_valid = np.load("data/x_valid.npy")
y_valid = np.load("data/y_valid.npy")
x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")

batch_size = 16
num_classes = 3
epochs = 100

# class weights to handle class imbalance
class_weights = {0: 1 - np.count_nonzero(y_train == 0) / len(y_train),
                 1: 1 - np.count_nonzero(y_train == 1) / len(y_train),
                 2: 1 - np.count_nonzero(y_train == 2) / len(y_train)}

# 将整型标签转为onehot
y_train = utils.to_categorical(y_train, num_classes)
y_valid = utils.to_categorical(y_valid, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# %%

def generator(image, label):
    return (image, label), (label, image)


dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_train = dataset_train.map(generator, num_parallel_calls=16)
dataset_train = dataset_train.batch(batch_size)
dataset_train = dataset_train.prefetch(-1)

dataset_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
dataset_valid = dataset_valid.map(generator, num_parallel_calls=16)
dataset_valid = dataset_valid.batch(batch_size)
dataset_valid = dataset_valid.prefetch(-1)

dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset_test = dataset_test.map(generator, num_parallel_calls=16)
dataset_test = dataset_test.batch(batch_size)
dataset_test = dataset_test.prefetch(-1)

# %%

import time


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# %%

# The best model is selected based on the loss value on the validation set
filepath = "model/weights/weights-CapsNet-best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_CapsNet_loss',
                             save_best_only=True, save_weights_only=True, verbose=1, mode='min')


# learning decay rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# learning schedule callback
lrate = LearningRateScheduler(step_decay)

time_callback = TimeHistory()

callbacks_list = [checkpoint, lrate, time_callback]

# %%

history = model.fit(dataset_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(dataset_valid),
                    # class_weight=class_weights,
                    shuffle=True,
                    callbacks=callbacks_list)

# %%

time_callback.totaltime

# %%

model.load_weights('model/weights/weights-CapsNet-best.h5')

# %%

predict = model.predict(dataset_test)
predict = predict[0]
y_pre = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)

report = classification_report(y_test, y_pre, output_dict=True)
df1 = pd.DataFrame(report).transpose()
# Write it into csv format
df1.to_csv('model/report.csv', index=True, header=True)
df1

# %%

perf_measures = compute_performance_measures(y_pre, y_test)
pm = {'acc': perf_measures.Acc,  # 准确率
      'recall': perf_measures.Recall,  # 召回率
      'spe': perf_measures.Specificity,  # 特异度
      'pre': perf_measures.Precision,  # 精确度
      'f1': perf_measures.F_measure,  # F1
      'avgacc': perf_measures.Overall_Acc
      }
df2 = pd.DataFrame(pm)
df2.to_csv('model/pm.csv', index=True, header=True)
df2

# %%

data = confusion_matrix(y_test, y_pre)
names = ['normal', 'pneumonia', 'COVID-19']
df_cm = pd.DataFrame(data, columns=names, index=names)
df_cm.to_csv('model/cm.csv', index=True, header=True)
df_cm

# %%

from sklearn.manifold import TSNE

layer_model = Model(inputs=model.input, outputs=model.get_layer('CapsNet').get_layer('digit_caps').output)
# 以这个model的预测值作为输出
feature = layer_model.predict(x_test)

feature_flattened = [a.flatten() for a in feature]

tsne = TSNE(n_components=2, random_state=0)

# prjected 2d data

feature_2d = tsne.fit_transform(feature_flattened)

feature_2d_covid = feature_2d[y_test == 2]
feature_2d_normal = feature_2d[y_test == 0]
feature_2d_non = feature_2d[y_test == 1]

plt.figure(figsize=(6, 4))
colors = ['mediumseagreen', 'cornflowerblue', 'darkorange']
classes = ['Normal', 'Pneumonia', 'COVID-19']

plt.scatter(feature_2d_covid[:, 0], feature_2d_covid[:, 1], c=colors[2], marker='o', label=classes[2])
plt.scatter(feature_2d_normal[:, 0], feature_2d_normal[:, 1], c=colors[0], marker='o', label=classes[0])
plt.scatter(feature_2d_non[:, 0], feature_2d_non[:, 1], c=colors[1], marker='o', label=classes[1])

# plt.title('COVID-19 t-SNE')
# plt.legend(loc="lower left")
plt.legend()
# plt.savefig('model/MHACapsNet t-SNE.png', dpi=500)
plt.show()

# %%