from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from tensorflow.keras.callbacks import LearningRateScheduler
#import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import scipy
from numpy import interp

import math
from itertools import cycle
import numpy as np
import pandas as pd
import random
import seaborn as sn
from matplotlib import pyplot as plt

from evaluation import compute_performance_measures

from utils_AAM import *
from layers_AAM import *
import  keras

#from 代码.layers import get_hvc_from_zxy_batch_norm_variable, caps_from_conv_zxy, hvc_from_zxy
#from 代码.对比.utils import margin_loss

K.set_image_data_format('channels_last')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  #按需分配显存
session = tf.compat.v1.Session(config=config)

#tf.random.set_seed(111111)
#np.random.seed(111111)
#random.seed(111111)

tf.__version__




# %%

def create_model(inputs, name='Norouting'):
    global model
    if name == 'Norouting':
        x = Conv2D(3, 7, 1, kernel_initializer='he_normal',groups=3)(inputs)
        x = Conv2D(16, 1, 1, kernel_initializer='he_normal')(x)
        # x = Conv2D(16, 5, 1, kernel_initializer='he_normal', dilation_rate=3,groups=16)(x)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)
        x = MaxPool2D(2)(x)

        x = Conv2D(16, 7, 1, kernel_initializer='he_normal',groups=16)(x)
        x = Conv2D(32, 1, 1, kernel_initializer='he_normal')(x)
        # x = Conv2D(32, 5, 1, kernel_initializer='he_normal', dilation_rate=3, groups=32)(x)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)
        x = MaxPool2D(2)(x)
        x = Conv2D(32, 7, 1, kernel_initializer='he_normal',groups=32)(x)
        x = Conv2D(64, 1, 1, kernel_initializer='he_normal')(x)
        # x = Conv2D(64, 5, 1, kernel_initializer='he_normal', dilation_rate=3, groups=64)(x)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)
        # x = AvgPool2D(2)(x)
        x = Conv2D(64, 7, 1, kernel_initializer='he_normal',groups=64)(x)
        x = Conv2D(128, 1, 1, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)
        # x = Conv2D(128, 5, 1, kernel_initializer='he_normal', dilation_rate=3, groups=128)(x)
        x = AAM(x, int(x.shape.as_list()[-1]), de=1, scope='AAM', trainable=True, reuse=False)
        
        a = x.shape[1]
        b = x.shape[3]
        print("a:", a)
        print("b:", b)

    else:
        x = Conv2D(16, 5, 1, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)

        x = MaxPool2D(2)(x)
        # x = AvgPool2D(2)(x)

        x = Conv2D(32, 5, 1, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)

        x = MaxPool2D(2)(x)
        # x = AvgPool2D(2)(x)

        x = Conv2D(64, 5, 2, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)

        x = Conv2D(128, 5, 1, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                               gamma_initializer="random_uniform")(x)
        x = LeakyReLU()(x)
        x = PrimaryCaps(256, 128, 9, 32, 8)(x)
    if name=='Norouting':
        x = caps_from_conv_zxy(x,(a*a*b)//8,8)
        variable=get_hvc_from_zxy_batch_norm_variable((a*a*b)//8,3,8)
        x = hvc_from_zxy(False, x, variable)
        digit_caps_len = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(x)
        print(digit_caps_len)
        model = Model(inputs=[inputs], outputs=[digit_caps_len], name=name)
      

    if name == 'MHACapsNet':
        digit_caps = MHACaps(3, 16, 2)(x)
        digit_caps_len = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(digit_caps)
        model = Model(inputs=[inputs], outputs=[digit_caps_len], name=name)

    if name == 'CapsNet':
        x = Reshape((1, 1, 32, 8))(x)
        digit_caps = DigitCaps(3, 16, 5)(x)
        digit_caps_len = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(digit_caps)
        model = Model(inputs=[inputs], outputs=[digit_caps_len], name=name)

    if name == 'ARCapsNet':
        x = Reshape((1, 1, 32, 8))(x)
        digit_caps = ARCaps(3, 16)(x)
        digit_caps_len = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(digit_caps)
        model = Model(inputs=[inputs], outputs=[digit_caps_len], name=name)

    #     if name == 'FCC':
    #         x = Reshape((1, 1, 32, 8))(x)
    #         digit_caps = DigitCaps(3, 16, None)(x)
    #         digit_caps_len = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(digit_caps)
    #         model = Model(inputs=[inputs], outputs=[digit_caps_len], name=name)
    if name == "MyFullyConnected":
      x = Conv2D(3, 7, 1, kernel_initializer='he_normal', padding='Same', groups=3)(inputs)
      x = Conv2D(16, 1, 1, kernel_initializer='he_normal')(x)
      x = Conv2D(16, 7, 1, kernel_initializer='he_normal', dilation_rate=3, groups=16)(x)

      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)
      x = MaxPool2D(2)(x)

      x = Conv2D(16, 7, 1, kernel_initializer='he_normal', padding='Same', groups=16)(x)
      x = Conv2D(32, 1, 1, kernel_initializer='he_normal')(x)
      x = Conv2D(32, 7, 1, kernel_initializer='he_normal', dilation_rate=3, groups=32)(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = Conv2D(32, 7, 1, kernel_initializer='he_normal', padding='Same', groups=32)(x)
      x = Conv2D(64, 1, 1, kernel_initializer='he_normal')(x)
      x = Conv2D(64, 7, 1, kernel_initializer='he_normal', dilation_rate=3, groups=64)(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = Conv2D(64, 7, 1, kernel_initializer='he_normal', padding='Same', groups=64)(x)
      x = Conv2D(128, 1, 1, kernel_initializer='he_normal')(x)
      x = Conv2D(128, 7, 1, kernel_initializer='he_normal', dilation_rate=3, groups=128)(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)
      x = GlobalAveragePooling2D(name='final_pooling')(x)
      x = Dense(3, activation='softmax', name='fc')(x)
      model = Model(inputs=inputs, outputs=x)

    if name =='CommonConvRouting':
      x = Conv2D(16, 5, 1, kernel_initializer='he_normal')(inputs)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = MaxPool2D(2)(x)
      # x = AvgPool2D(2)(x)

      x = Conv2D(32, 5, 1, kernel_initializer='he_normal')(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = MaxPool2D(2)(x)
      # x = AvgPool2D(2)(x)

      x = Conv2D(64, 5, 2, kernel_initializer='he_normal')(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = Conv2D(128, 5, 1, kernel_initializer='he_normal')(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)
      a = x.shape[1]
      b = x.shape[3]
      x = caps_from_conv_zxy(x, (a * a * b) //16, 16)
      variable = get_hvc_from_zxy_batch_norm_variable((a * a * b) //16, 3, 16)
      x = hvc_from_zxy(False, x, variable)
      # x= tf.norm(x, axis=2)
      digit_caps_len = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(x)
      print(digit_caps_len)
      model = Model(inputs=[inputs], outputs=[digit_caps_len], name=name)
    
    if name == "CommonFullyConnected":
      x = Conv2D(16, 7, 1, kernel_initializer='he_normal')(inputs)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = MaxPool2D(2)(x)
      # x = AvgPool2D(2)(x)

      x = Conv2D(32, 7, 1, kernel_initializer='he_normal')(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = MaxPool2D(2)(x)
      # x = AvgPool2D(2)(x)

      x = Conv2D(64, 7, 2, kernel_initializer='he_normal')(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)

      x = Conv2D(128, 7, 1, kernel_initializer='he_normal')(x)
      x = BatchNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(x)
      x = LeakyReLU()(x)
      x = GlobalAveragePooling2D(name='final_pooling')(x)
      x = Dense(3, activation='softmax', name='fc')(x)
      model = Model(inputs=inputs, outputs=x)

    return model


# %%

inputs = Input(shape=(128, 128, 3))
model = create_model(inputs,'Norouting')
adam = optimizers.Adam(lr=0.0001)
model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])
model.summary()

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

# learning decay rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# %%

batch_size = 128    #batch_size:一次训练所抓取的数据样本数量
num_classes = 3    #num_classes：标签类别个数
epochs = 50       #epochs:周期

images = np.load("newData/image.npy")
labels = np.load("newData/label.npy")

# images= np.load("newData/image.npy")
# labels= np.load("newData/label.npy")

# x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=3)

# %%

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# define 4-fold cross validation test harness
kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=0)
cvscores = []
cvpre = []
cvrecall = []
cvf1 = []
cvauc = []


for k, (train, test) in enumerate(kfold.split(images, labels)):
  if k==0:
    path = "model-cv/cv" + str(k + 1) + "/"

    x_train = images[train]
    x_test = images[test]
    y_train = labels[train]
    y_test = labels[test]

    np.save(path + 'data/x_train.npy', x_train)  #np.save()可以将数组保存为 .npy 格式的二进制文件
    np.save(path + 'data/y_train.npy', y_train)
    np.save(path + 'data/x_test.npy', x_test)
    np.save(path + 'data/y_test.npy', y_test)

    # class weights to handle class imbalance  用于处理类不平衡的类权重
    class_weights = {0: 1 - np.count_nonzero(y_train == 0) / len(y_train),    #numpy.count_nonzero是用于统计数组中非零元素的个数
                     1: 1 - np.count_nonzero(y_train == 1) / len(y_train),
                     2: 1 - np.count_nonzero(y_train == 2) / len(y_train)}

    # 将整型标签转为onehot
    y_train = utils.to_categorical(y_train, num_classes)#(795,3)
    y_test = utils.to_categorical(y_test, num_classes)#(199,3)

    # The best model is selected based on the loss value on the validation set
    filepath = path + "weights-best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_weights_only=True, save_best_only=True, mode='max')

    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)

    time_callback = TimeHistory()
    callbacks_list = [checkpoint, lrate, time_callback]

    inputs = Input(shape=(128, 128, 3))

    # ===================================Model======================================
    model = create_model(inputs, name="Norouting")  # , name="ARCapsNet")#, name="CapsNet")

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    #model.summary()
    print("========================= 第" + str(k + 1) + "折开始 ============================")
    print('x train:',type(x_train),x_train.shape,y_train.shape)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        # validation_split=0.1,
                        validation_data=(x_test, y_test),
                        class_weight=class_weights,
                        shuffle=True,
                        callbacks=callbacks_list)


    np.save(path + 'acc.npy', history.history['accuracy'])
    np.save(path + 'val_acc.npy', history.history['val_accuracy'])
    np.save(path + 'loss.npy', history.history['loss'])
    np.save(path + 'val_loss.npy', history.history['val_loss'])

    model.load_weights(filepath)
    # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    cvscores.append(scores[1] * 100)

    predict = model.predict([x_test])

    # ===================auc=========================
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predict[:, i], )
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_auc["macro"] = roc_auc_score(y_test, predict, multi_class="ovo", average="macro")
    roc_auc["weighted"] = roc_auc_score(y_test, predict, multi_class="ovo", average="weighted")
    # ===================================================

    y_pre = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)

    report = classification_report(y_test, y_pre, output_dict=True)
    df1 = pd.DataFrame(report).transpose()

    df1['auc'] = [roc_auc[0], roc_auc[1], roc_auc[2], " ", roc_auc["macro"], roc_auc["weighted"]]
    df1.to_csv('./model-cv/fig/' + 'data2_report.csv', index=True, header=True)

    cvpre.append(df1.loc['macro avg', 'precision'] * 100)


    cvrecall.append(df1.loc['macro avg', 'recall'] * 100)
    cvf1.append(df1.loc['macro avg', 'f1-score'] * 100)
    cvauc.append(df1.loc['macro avg', 'auc'] * 100)

    data = confusion_matrix(y_test, y_pre)
    names = ['normal', 'pneumonia', 'COVID-19']
    df_cm = pd.DataFrame(data, columns=names, index=names)
    df_cm.to_csv(path + 'cm.csv', index=True, header=True)
    print("========================= 第" + str(k + 1) + "折结束 ============================")


df2 = pd.DataFrame(index=['Fold1'])
df2['acc'] =cvscores
df2['pre']=cvpre
df2['cvrecall']=cvrecall
df2['cvf1']=cvf1
df2['cvauc']=cvauc
for i in range(5):
  if i==0:
    df2.loc[4,['acc']]=np.mean(cvscores)
    df2.loc[5,['acc']]=np.std(cvscores)
  if i==1:
    df2.loc[4,['pre']]=np.mean(cvpre)
    df2.loc[5,['pre']]=np.std(cvpre)
  if i==2:
    df2.loc[4,['cvrecall']]=np.mean(cvrecall)
    df2.loc[5,['cvrecall']]=np.std(cvrecall)
  if i==3:
    df2.loc[4,['cvf1']]=np.mean(cvf1)
    df2.loc[5,['cvf1']]=np.std(cvf1)
  else:
    df2.loc[4,['cvauc']]=np.mean(cvauc)
    df2.loc[5,['cvauc']]=np.std(cvauc)

df2.to_csv('./model-cv/fig/data2_fold1_result.csv', index=True, header=True)

print("accuracy：%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("precision：%.2f%% (+/- %.2f%%)" % (np.mean(cvpre), np.std(cvpre)))
print("recall：%.2f%% (+/- %.2f%%)" % (np.mean(cvrecall), np.std(cvrecall)))
print("f1-score：%.2f%% (+/- %.2f%%)" % (np.mean(cvf1), np.std(cvf1)))
print("auc：%.2f%% (+/- %.2f%%)" % (np.mean(cvauc), np.std(cvauc)))

#sum(time_callback.times)/ 100




path1 = './model-cv/cv1/'
path2 = './model-cv/cv2/'
path3 = './model-cv/cv3/'
path4 = './model-cv/cv4/'



# %%
import warnings
warnings.filterwarnings("ignore")
#model.save_weights(path1 + 'weights-best.h5')
model.load_weights(path1 + 'weights-best.h5')


def load_img_mai(IMAGE_PATH):
  img = tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(128, 128))
  img = tf.keras.preprocessing.image.img_to_array(img)
  x1 = np.expand_dims(img, axis=0)
  x1=x1.astype('float32')/255
  return x1

def load_img(IMAGE_PATH):
  return tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(128, 128))


#data2
COVID=['./newData/data1/covid-19/COVID (454).png',
    './newData/data1/covid-19/COVID (468).png','./newData/data1/covid-19/COVID (668).png',
    './newData/data1/covid-19/COVID (793).png',
    './newData/data1/covid-19/COVID (1149).png','./newData/data1/covid-19/COVID (1145).png',
    './newData/data1/covid-19/COVID (60).png','./newData/data1/covid-19/COVID (130).png',
    './newData/data1/covid-19/COVID (105).png','./newData/data1/covid-19/COVID (121).png',
    './newData/data1/covid-19/COVID (133).png','./newData/data1/covid-19/COVID (394).png',
    './newData/data1/covid-19/COVID (436).png','./newData/data1/covid-19/COVID (458).png',
    './newData/data1/covid-19/COVID (468).png','./newData/data1/covid-19/COVID (530).png',
    './newData/data1/covid-19/COVID (537).png','./newData/data1/covid-19/COVID (601).png',
    './newData/data1/covid-19/COVID (665).png','./newData/data1/covid-19/COVID (670).png',
    './newData/data1/covid-19/COVID (689).png','./newData/data1/covid-19/COVID (785).png',
    './newData/data1/covid-19/COVID (1101).png','./newData/data1/covid-19/COVID (794).png',
    './newData/data1/covid-19/COVID (1102).png','./newData/data1/covid-19/COVID (394).png',
    './newData/data1/covid-19/COVID (786).png','./newData/data1/covid-19/COVID (451).png',
    './newData/data1/covid-19/COVID (341).png','./newData/data1/covid-19/COVID (530).png',
    './newData/data1/covid-19/COVID (401).png','./newData/data1/covid-19/COVID (665).png',
    './newData/data1/covid-19/COVID (402).png','./newData/data1/covid-19/COVID (669).png',
    './newData/data1/covid-19/COVID (451).png','./newData/data1/covid-19/COVID (830).png',

    ]


pneumonia=['./newData/data1/viral pneumonia/Viral Pneumonia (190).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (268).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (332).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (358).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (412).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (482).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (553).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (617).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (779).png',
      './newData/data1/viral pneumonia/Viral Pneumonia (1058).png',
      ]

#data1

# COVID=['./data/data1/covid-19/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1a.png',
#     './data/data1/covid-19/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-002-fig3a.png',
#     './data/data1/covid-19/1-s2.0-S2531043720300921-gr1.png',
#     './data/data1/covid-19/1-s2.0-S2387020620301959-gr4_lrg-c.png',
#     './data/data1/covid-19/2-chest-filmc.jpg',
#     './data/data1/covid-19/7-fatal-covid19.jpg',
#     './data/data1/covid-19/16865_1_1.jpg',
#     './data/data1/covid-19/article_river_c79329e06dff11eab69c95940c7c0d00-CXR-D1-COVID19.png',
#     './data/data1/covid-19/article_river_e4d185c06e3511eaa2321d8ab357a1de-c1mn.png',
#     './data/data1/covid-19/radiol.2020201160.fig2a.jpeg',
#     './data/data1/covid-19/radiol.2020201160.fig3c.jpeg',]


# pneumonia=['./data/data1/viral pneumonia/person49_bacteria_235.jpeg',
#       './data/data1/viral pneumonia/person23_bacteria_105.jpeg',
#       './data/data1/viral pneumonia/person61_bacteria_291.jpeg',
#       './data/data1/viral pneumonia/person278_virus_574.jpeg',
#       './data/data1/viral pneumonia/person448_bacteria_1936.jpeg',
#       './data/data1/viral pneumonia/person724_virus_1343.jpeg',
#       './data/data1/viral pneumonia/person825_bacteria_2736.jpeg',
#       './data/data1/viral pneumonia/person1484_bacteria_3878.jpeg',
#       './data/data1/viral pneumonia/person1817_bacteria_4675.jpeg',
#       './data/data1/viral pneumonia/person979_virus_1654.jpeg',]

from tf_keras_vis.utils.scores import CategoricalScore
from vis.utils import utils
from tf_keras_vis.saliency import Saliency
from matplotlib import cm

score_COVID = CategoricalScore([2])
score_Pneumonia = CategoricalScore([1])

saliency = Saliency(model)

#保存图片
for i in range(len(COVID)):
    COVID_img=load_img(COVID[i])
    plt.figure(figsize=(6,6))
    plt.imshow(COVID_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./CAM/COVID-19/'+str(i)+'.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
    plt.close()

# for i in range(len(pneumonia)):
#     COVID_img=load_img(pneumonia[i])
#     plt.figure(figsize=(6,6))
#     plt.imshow(COVID_img)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/Pneumonia/'+str(i)+'.pdf',bbox_inches='tight',pad_inches =0,dpi=1000)
#     plt.close()

#saliency图



# for i in range(len(COVID)):
#     COVID_img=load_img(COVID[i])
#     COVID_process = load_img_mai(COVID[i])
#     saliency_map = saliency(score_COVID, COVID_process)
#     saliency_map = np.uint8(cm.jet(saliency_map)[..., :3] * 255)
#     plt.figure(figsize=(6,6))
#     plt.imshow(COVID_img)
#     plt.imshow(saliency_map[0], cmap='jet', alpha=0.5)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/COVID-19/'+str(i)+'_saliency_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
#     plt.close()

# for i in range(len(pneumonia)):
#     pneumonia_img=load_img(pneumonia[i])
#     pneumonia_process = load_img_mai(pneumonia[i])
#     saliency_map = saliency(score_Pneumonia, pneumonia_process)
#     saliency_map = np.uint8(cm.jet(saliency_map)[..., :3] * 255)
#     plt.figure(figsize=(6,6))
#     plt.imshow(pneumonia_img)
#     plt.imshow(saliency_map[0], cmap='jet', alpha=0.5)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/Pneumonia/'+str(i)+'_saliency_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
#     plt.close()

#Scorecam
from tf_keras_vis.scorecam import Scorecam
layer_index = utils.find_layer_idx(model, 'tf.linalg.matmul_5')
Scorecam=Scorecam(model)
for i in range(len(COVID)):
    COVID_img=load_img(COVID[i])
    COVID_process = load_img_mai(COVID[i])
    Scorecam_map = Scorecam(score_COVID, COVID_process,penultimate_layer=layer_index)
    Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
    plt.figure(figsize=(6,6))
    plt.imshow(COVID_img)
    plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./CAM/COVID-19/'+str(i)+'_Scorecam_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
    plt.close()

# for i in range(len(pneumonia)):
#     pneumonia_img=load_img(pneumonia[i])
#     pneumonia_process = load_img_mai(pneumonia[i])
#     Scorecam_map = Scorecam(score_Pneumonia, pneumonia_process,penultimate_layer=layer_index)
#     Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
#     plt.figure(figsize=(6,6))
#     plt.imshow(pneumonia_img)
#     plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/Pneumonia/'+str(i)+'_Scorecam_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
#     plt.close()

#Layercam
from tf_keras_vis.layercam import Layercam
Layercam=Layercam(model)
for i in range(len(COVID)):
    COVID_img=load_img(COVID[i])
    COVID_process = load_img_mai(COVID[i])
    Scorecam_map = Layercam(score_COVID, COVID_process,penultimate_layer=layer_index)
    Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
    plt.figure(figsize=(6,6))
    plt.imshow(COVID_img)
    plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./CAM/COVID-19/'+str(i)+'_Layercam_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
    plt.close()

# for i in range(len(pneumonia)):
#     pneumonia_img=load_img(pneumonia[i])
#     pneumonia_process = load_img_mai(pneumonia[i])
#     Scorecam_map = Layercam(score_Pneumonia, pneumonia_process,penultimate_layer=layer_index)
#     Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
#     plt.figure(figsize=(6,6))
#     plt.imshow(pneumonia_img)
#     plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/Pneumonia/'+str(i)+'_Layercam_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
#     plt.close()

#GrandCam
# from tf_keras_vis.gradcam import Gradcam
# Layercam=Gradcam(model)
# for i in range(len(COVID)):
#     COVID_img=load_img(COVID[i])
#     COVID_process = load_img_mai(COVID[i])
#     Scorecam_map = Layercam(score_COVID, COVID_process,penultimate_layer=layer_index)
#     Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
#     plt.figure(figsize=(6,6))
#     plt.imshow(COVID_img)
#     plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/COVID-19/'+str(i)+'_Gradcam_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
#     plt.close()

# for i in range(len(pneumonia)):
#     pneumonia_img=load_img(pneumonia[i])
#     pneumonia_process = load_img_mai(pneumonia[i])
#     Scorecam_map = Layercam(score_Pneumonia, pneumonia_process,penultimate_layer=layer_index)
#     Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
#     plt.figure(figsize=(6,6))
#     plt.imshow(pneumonia_img)
#     plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/Pneumonia/'+str(i)+'_Gradcam_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
#     plt.close()

#Gradcam++
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
Layercam=GradcamPlusPlus(model)
for i in range(len(COVID)):
    COVID_img=load_img(COVID[i])
    COVID_process = load_img_mai(COVID[i])
    Scorecam_map = Layercam(score_COVID, COVID_process,penultimate_layer=layer_index)
    Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
    plt.figure(figsize=(6,6))
    plt.imshow(COVID_img)
    plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./CAM/COVID-19/'+str(i)+'_gradcam_plus_plus_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
    plt.close()

# for i in range(len(pneumonia)):
#     pneumonia_img=load_img(pneumonia[i])
#     pneumonia_process = load_img_mai(pneumonia[i])
#     Scorecam_map = Layercam(score_Pneumonia, pneumonia_process,penultimate_layer=layer_index)
#     Scorecam_map = np.uint8(cm.jet(Scorecam_map)[..., :3] * 255)
#     plt.figure(figsize=(6,6))
#     plt.imshow(pneumonia_img)
#     plt.imshow(Scorecam_map[0], cmap='jet', alpha=0.5)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./CAM/Pneumonia/'+str(i)+'_gradcam_plus_plus_map.pdf',dpi=1000,bbox_inches='tight',pad_inches =0)
#     plt.close()

print('-----CAM Finish-------')

# #加载图片
# img1=load_img(Normal)
# img2=load_img(pneumonia)
# img3=load_img(COVID)

# #处理过后的图片
# Normal=load_img_mai(Normal)
# pneumonia=load_img_mai(pneumonia)
# COVID=load_img_mai(COVID)

# print('--'*20)
# print(np.argmax(model.predict(Normal)))
# print(np.argmax(model.predict(pneumonia)))
# print(np.argmax(model.predict(COVID)))

# #替换softmax
# from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
# replace2linear = ReplaceToLinear()
# from tf_keras_vis.utils.scores import CategoricalScore
# score1 = CategoricalScore([0])
# score2 = CategoricalScore([1])
# score3 = CategoricalScore([2])

# #绘制Vanilla Saliency显著性图
# from tf_keras_vis.scorecam import Scorecam
# layer_index = utils.find_layer_idx(model, 'conv2d_22')
# # from tf_keras_vis.saliency import Saliency
# saliency = Scorecam(model)
# saliency_map1 = saliency(score1,Normal,penultimate_layer=layer_index)
# saliency_map2 = saliency(score2,pneumonia,penultimate_layer=layer_index)
# saliency_map3 = saliency(score3,COVID,penultimate_layer=layer_index)

# #GrandCam
# from vis.utils import utils
# from matplotlib import cm
# from tf_keras_vis.gradcam import Gradcam

# gradcam = Scorecam(model)
# layer_index = utils.find_layer_idx(model, 'conv2d_23')

# # Generate heatmap with GradCAM
# Gcam1 = gradcam(score1,Normal,penultimate_layer=layer_index)
# Gcam2 = gradcam(score2,pneumonia,penultimate_layer=layer_index)
# Gcam3 = gradcam(score3,COVID,penultimate_layer=layer_index)

# #绘制GradCAM++
# layer_index = utils.find_layer_idx(model, 'conv2d_24')
# from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
# gradcam = Scorecam(model)
# Gcam_1 = gradcam(score1,Normal,penultimate_layer=layer_index)
# Gcam_2 = gradcam(score2,pneumonia,penultimate_layer=layer_index)
# Gcam_3 = gradcam(score3,COVID,penultimate_layer=layer_index)


# #ScoreCAM
# from tf_keras_vis.scorecam import Scorecam
# layer_index = utils.find_layer_idx(model, 'conv2d_25')
# scorecam = Scorecam(model)
# scorecam1 = scorecam(score1,Normal,penultimate_layer=layer_index)
# scorecam2 = scorecam(score2,pneumonia,penultimate_layer=layer_index)
# scorecam3 = scorecam(score3,COVID,penultimate_layer=layer_index)

# #LayerCAM
# from tf_keras_vis.layercam import Layercam
# layer_index = utils.find_layer_idx(model, 'leaky_re_lu_7')
# Layercam = Scorecam(model)
# Layercam1 = Layercam(score1,Normal, penultimate_layer=layer_index)
# Layercam2 = Layercam(score2,pneumonia, penultimate_layer=layer_index)
# Layercam3 = Layercam(score3,COVID, penultimate_layer=layer_index)

# plt.subplots(nrows=3, ncols=6, figsize=(14, 7))
# plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=-0.1,hspace=-0.1)

# #第一行
# plt.subplot(361)
# plt.imshow(img1)
# plt.xticks([])
# plt.yticks([])
# plt.ylabel(image_titles[0], fontsize=10)
# plt.title("COVID-19",fontsize=12)
# plt.axis('off')

# plt.subplot(362)
# saliency_map1 = np.uint8(cm.jet(saliency_map1)[..., :3] * 255)
# plt.title("Depthwise Convolution",fontsize=12)
# plt.imshow(img1)
# plt.imshow(saliency_map1[0], cmap='jet',alpha=0.5)
# plt.axis('off')

# plt.subplot(363)
# heatmap = np.uint8(cm.jet(Gcam1)[..., :3] * 255)
# plt.title("Point Convolution", fontsize=12)
# plt.imshow(img1)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(364)
# heatmap = np.uint8(cm.jet(Gcam_1)[..., :3] * 255)
# plt.title("Dilated Convolution", fontsize=12)
# plt.imshow(img1)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(365)
# heatmap = np.uint8(cm.jet(scorecam1)[..., :3] * 255)
# plt.title("Deformed Convolution", fontsize=12)
# plt.imshow(img1)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(366)
# heatmap = np.uint8(cm.jet(Layercam1)[..., :3] * 255)
# plt.title("ASA", fontsize=12)
# plt.imshow(img1)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# #第2行
# plt.subplot(367)
# plt.imshow(img2)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')

# plt.subplot(368)
# plt.imshow(img2)
# saliency_map2 = np.uint8(cm.jet(saliency_map2)[..., :3] * 255)
# plt.imshow(saliency_map2[0], cmap='jet')
# plt.axis('off')

# plt.subplot(369)
# heatmap = np.uint8(cm.jet(Gcam2)[..., :3] * 255)
# plt.imshow(img2)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(3,6,10)
# heatmap = np.uint8(cm.jet(Gcam_2)[..., :3] * 255)
# plt.imshow(img2)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(3,6,11)
# heatmap = np.uint8(cm.jet(scorecam2)[..., :3] * 255)
# plt.imshow(img2)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(3,6,12)
# heatmap = np.uint8(cm.jet(Layercam2)[..., :3] * 255)
# plt.imshow(img2)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# #第3行
# plt.subplot(3,6,13)
# plt.imshow(img3)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')

# plt.subplot(3,6,14)
# saliency_map3 = np.uint8(cm.jet(saliency_map3)[..., :3] * 255)
# plt.imshow(img3)
# plt.imshow(saliency_map3[0], cmap='jet')
# plt.axis('off')

# plt.subplot(3,6,15)
# heatmap = np.uint8(cm.jet(Gcam3)[..., :3] * 255)
# plt.imshow(img3)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.6) # overlay
# plt.axis('off')

# plt.subplot(3,6,16)
# heatmap = np.uint8(cm.jet(Gcam_3)[..., :3] * 255)
# plt.imshow(img3)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(3,6,17)
# heatmap = np.uint8(cm.jet(scorecam3)[..., :3] * 255)
# plt.imshow(img3)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.subplot(3,6,18)
# heatmap = np.uint8(cm.jet(Layercam3)[..., :3] * 255)
# plt.imshow(img3)
# plt.imshow(heatmap[0], cmap='jet', alpha=0.5) # overlay
# plt.axis('off')

# plt.tight_layout()
# plt.savefig("./CAM/saliency_map.pdf",dpi=1000)
# #plt.show()



# %%
from tensorflow.keras import utils
x_test = np.load(path1 + 'data/x_test.npy')
y_test = np.load(path1 + 'data/y_test.npy')

y_test = utils.to_categorical(y_test, num_classes)

predict = model.predict([x_test])

# ===================auc=========================
n_classes = y_test.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predict[:, i], )
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc["macro"] = roc_auc_score(y_test, predict, multi_class="ovo", average="macro")
roc_auc["weighted"] = roc_auc_score(y_test, predict, multi_class="ovo", average="weighted")
# ===================================================

y_pre = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)

report = classification_report(y_test, y_pre, output_dict=True)
df1 = pd.DataFrame(report).transpose()

df1['auc'] = [roc_auc[0], roc_auc[1], roc_auc[2], " ", roc_auc["macro"], roc_auc["weighted"]]

# Write it into csv format
# print(df1)

'''
plt.figure(figsize=(12, 8))
color = ['r', 'r', 'r']
a = y_test[y_pre != y_test]
for i ,(img , p) in enumerate(zip(x_test[y_pre != y_test], predict[y_pre != y_test])):
    if (i<=3):
        print(i)
        ax1 = plt.subplot(2,4,i+1)
        ax1.imshow(img)
        #ax.set_title("label = %d"%label)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2 = plt.subplot(2,4,i+5)
        color[a[i]] = 'b'
        ax2.bar(x=['Normal', 'Pneumonia', 'Covid-19'], height=p, width=0.5, color=color)
        # ax.set_title("label = %d"%label)
        ax2.set_xticklabels(['Normal', 'Pneumonia', 'Covid-19'],
                            rotation=30,  #fontsize = 'large',
                            fontdict={'fontsize': 13})
        plt.yticks(fontsize=13)
        color = ['r', 'r', 'r']
plt.tight_layout()
plt.savefig('./model-cv/fig/failure1.pdf',dpi=1000)
plt.show()'''


plt.figure(figsize=(12,6))
color=['r','r','r']
a = y_test[y_pre!=y_test]
for i,(img, p) in enumerate(zip(x_test[y_pre!=y_test], predict[y_pre!=y_test])):
     if(i<=3):
         ax1=plt.subplot(2,4,i+1)
         ax1.imshow(img)
         #ax.set_title("label = %d"%label)
         ax1.set_xticks([])
         ax1.set_yticks([])
         ax2=plt.subplot(2,4,i+5)
         color[a[i]] = 'b'
         ax2.bar(x=['Normal','Pneumonia','Covid-19'],height=p,width=0.5,color=color)
         #ax.set_title("label = %d"%label)
         color=['r','r','r']
plt.tight_layout()
plt.savefig('./model-cv/fig/data2_failure.pdf',dpi=1000)
plt.show()


# 将标签二值化
y = label_binarize(y_test, classes=[0, 1, 2])
y_pre = predict
# 设置种类
n_classes = y.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pre[:, i], )
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pre.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
roc_auc["micro"] = roc_auc_score(y, y_pre, multi_class="ovo", average="micro")

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
roc_auc["macro"] = roc_auc_score(y, y_pre, multi_class="ovo", average="macro")

# %%

# Plot all ROC curves
plt.rcParams.update({'font.size': 13})
plt.figure(figsize=(7, 5), dpi=1000)
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'.format(roc_auc["micro"]), color=[0,153/255,153/255], linestyle=':',
         linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'.format(roc_auc["macro"]), color=[204/255,0,102/255], linestyle=':',
         linewidth=4)

colors = cycle(['blue', 'green', 'red'])
classes = ['Normal', 'Pneumonia', 'COVID-19']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle='-',
             label='ROC curve of class {0} (area = {1:0.4f})'.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right", fontsize='x-small')
# {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'})
plt.tight_layout()
plt.savefig('./model-cv/fig/data2_ROC-fold1.pdf', dpi=1000)
plt.show()

acc = np.load(path1 + "acc.npy")
val_acc = np.load(path1 + "val_acc.npy")
loss = np.load(path1 + "loss.npy")
val_loss = np.load(path1 + "val_loss.npy")
#
#
#
# plt.style.use(['science', 'ieee'])
#
#
#
#
with plt.style.context(['ieee']):
    fig = plt.figure(figsize=(3.5, 3.5), dpi=1000)
    # 布局与图例
    layout = (2, 1)
    acc_ax = plt.subplot2grid(layout, (0, 0))
    loss_ax = plt.subplot2grid(layout, (1, 0))
    acc_ax.plot(acc, color='blue')
    acc_ax.plot(val_acc)
    acc_ax.set_title('Accuracy vs. Number of Training Epochs')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.set_xlabel('Epochs')
    acc_ax.legend(['Training', 'Validation'])

    loss_ax.plot(loss, color='blue')
    loss_ax.plot(val_loss)
    loss_ax.set_title('Loss vs. Number of Training Epochs')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_xlabel('Epochs')
    loss_ax.legend(['Training', 'Validation'])

    # 自动调整图例布局
    plt.tight_layout()
    plt.savefig('./model-cv/fig/data2_fold1_acc+loss.pdf', dpi=1000)
    plt.show()
#
#
#
X = np.load("data/image.npy")
Y = np.load("data/label.npy")
#
predict = model.predict([X])
#
y = label_binarize(Y, classes=[0, 1, 2])
n_classes = y.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], predict[:, i], )
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc["macro"] = roc_auc_score(y, predict, multi_class="ovo", average="macro")
roc_auc["weighted"] = roc_auc_score(y, predict, multi_class="ovo", average="weighted")

y_pre = np.argmax(predict, axis=1)

report = classification_report(Y, y_pre, output_dict=True)
df1 = pd.DataFrame(report).transpose()
df1['auc'] = [roc_auc[0], roc_auc[1], roc_auc[2], '', roc_auc["macro"], roc_auc["weighted"]]
df1.to_csv('./model-cv/fig/data2_to_prefict_data1.csv', index=True, header=True)
#
#
#
#
#
# # 将标签二值化
y = label_binarize(Y, classes=[0, 1, 2])
y_pre = predict
#
# 设置种类
n_classes = y.shape[1]
#
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pre[:, i], )
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pre.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
roc_auc["micro"] = roc_auc_score(y, y_pre, multi_class="ovo", average="micro")

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
roc_auc["macro"] = roc_auc_score(y, y_pre, multi_class="ovo", average="macro")
#
# # Plot all ROC curves
plt.rcParams.update({'font.size': 13})
plt.figure(figsize=(7, 5))
lw = 2
#
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':',
         linewidth=4)
#
plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (area = {0:0.4f})'.format(roc_auc["macro"]), color='navy', linestyle=':',
         linewidth=4)
#
colors = cycle(['blue', 'green', 'red'])
classes = ['Normal', 'Pneumonia', 'COVID-19']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle='-',
            label='ROC curve of class {0} (area = {1:0.4f})'.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right", fontsize='x-small')
# {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'})
plt.tight_layout()
plt.savefig('./model-cv/fig/data2_to_ROC-data1.pdf',dpi=1000)
# plt.show()
#
#
#
# yy_pred = model.predict(x_test) # 预测出[[0.4,0.45],[0.8,0.3],[0.6,0.71]]
# y_pre = np.argmax(yy_pred, axis=1) # 选择max值进行输出0,或1
#
#y_pre = np.reshape(y_pre,[y_pre.shape[0]*y_pre.shape[1]])

Y = Y.astype(np.float)
y_pre = np.argmax(predict, axis=1)
#
data = confusion_matrix(Y, y_pre)
names = ['Normal', 'Pneumonia', 'COVID-19']
df_cm = pd.DataFrame(data, columns=names, index=names)
print(df_cm)
#
#
#
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig = plt.figure(figsize=(3.5, 3),dpi=1000)
#ax = plt.subplot(222)
sn.set(font_scale=0.4)  # for label size
h_m = sn.heatmap(df_cm, cmap="RdPu", annot=True, annot_kws={"size": 5, "fontweight": "bold"}, fmt='d', cbar=True)

h_m.set_yticklabels(h_m.get_yticklabels(), rotation=0, fontweight="bold")
h_m.set_xticklabels(h_m.get_xticklabels(), rotation=45, fontweight="bold")
plt.tight_layout()
fig.savefig('./model-cv/fig' + '/' + 'confusion_data1.pdf', dpi=1000)
plt.show()

##绘制数据集1的混淆矩阵
X = np.load(path1 + 'data/x_test.npy')
Y = np.load(path1 + 'data/y_test.npy')
predict = model.predict([X])

Y = Y.astype(np.float)
y_pre = np.argmax(predict, axis=1)
#
data = confusion_matrix(Y, y_pre)
names = ['Normal', 'Pneumonia', 'COVID-19']
df_cm = pd.DataFrame(data, columns=names, index=names)
print(df_cm)
#
#
#
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig = plt.figure(figsize=(3.5, 3),dpi=1000)
#ax = plt.subplot(222)
sn.set(font_scale=0.4)  # for label size
h_m = sn.heatmap(df_cm, cmap="RdPu", annot=True, annot_kws={"size": 5, "fontweight": "bold"}, fmt='d', cbar=True)

h_m.set_yticklabels(h_m.get_yticklabels(), rotation=0, fontweight="bold")
h_m.set_xticklabels(h_m.get_xticklabels(), rotation=45, fontweight="bold")
plt.tight_layout()
fig.savefig('./model-cv/fig' + '/' + 'train_confusion_data2.pdf', dpi=1000)

#
path_list = [path1, path2, path3, path4]
fig = plt.figure(figsize=(14, 3))
sn.set(font_scale=0.6)  # for label size
x = 140
for path in path_list:
    x += 1
    df_cm = pd.read_csv(path + 'cm.csv', index_col=0)
    #     df_cm.index.name = 'Actual'
    #     df_cm.columns.name = 'Predicted'
    ax = plt.subplot(x)
    h_m = sn.heatmap(df_cm, cmap="RdPu", annot=True, annot_kws={"size": 10, "fontweight": "bold"}, fmt='d', ax=ax,
                     cbar=True)
    h_m.set_yticklabels(h_m.get_yticklabels(), rotation=0, fontweight="bold")
    h_m.set_xticklabels(h_m.get_xticklabels(), rotation=0, fontweight="bold")
    h_m.set_title('Fold-' + str(x - 140), fontdict={'fontsize': 8, 'fontweight': "bold"})

# #自动调整图例布局
plt.tight_layout()
fig.savefig('./model-cv/fig/' + 'confusion_matrix_folds.pdf', dpi=1000)
plt.show()
